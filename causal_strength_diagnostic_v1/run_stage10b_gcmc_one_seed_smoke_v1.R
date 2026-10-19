args <- commandArgs(trailingOnly=TRUE)
record <- normalizePath(args[[1]], winslash="/", mustWork=TRUE); mode <- args[[2]]
local_lib <- "F:/GeoSIMCP_Reproduce/environment/gcmc_spedm_1.12_r4.6.1/library"
.libPaths(c(local_lib,.libPaths()))
stopifnot(as.character(packageVersion("spEDM")) == "1.12")
features <- c("x1","x2","g_proxy","x_noise")
scenarios <- c("S2","S4")
edges <- read.csv(file.path(record,"inputs","delaunay_edges.csv"),check.names=FALSE)
make_nb <- function(n, edges) {
  nb <- lapply(seq_len(n), function(i) sort(unique(c(edges$v[edges$u==i],edges$u[edges$v==i]))))
  nb <- lapply(nb, as.integer); class(nb) <- "nb"; attr(nb,"region.id") <- as.character(seq_len(n)); attr(nb,"sym") <- TRUE
  nb
}
nb <- make_nb(200,edges)
load_sf <- function(s) {
  d <- read.csv(file.path(record,"inputs",paste0(s,"_scale_fit_gcmc.csv")),check.names=FALSE)
  stopifnot(nrow(d)==200,!anyDuplicated(d$sample_id),all(is.finite(as.matrix(d[c("s_x","s_y",features,"A_model")]))))
  sf::st_as_sf(d,coords=c("s_x","s_y"),remove=TRUE,crs=NA)
}
datasets <- setNames(lapply(scenarios,load_sf),scenarios)
warns <- character(); capture_warn <- function(expr) withCallingHandlers(expr,warning=function(w){warns<<-c(warns,conditionMessage(w));invokeRestart("muffleWarning")})
run_one <- function(s, feature, E, k, style) {
  t0 <- proc.time()[[3]]
  z <- capture_warn(spEDM::gcmc(datasets[[s]],cause=feature,effect="A_model",libsizes=200,E=E,k=k,tau=1,style=style,dist.metric="L2",threads=1,detrend=FALSE,parallel.level="low",bidirectional=TRUE,progressbar=FALSE,nb=nb))
  stopifnot(identical(z$varname,c(feature,"A_model")),nrow(z$cs)==1,nrow(z$xmap)==1)
  data.frame(scenario=s,feature=feature,E=E,k=k,style=style,tau=1,forward_direction=paste0(feature," -> A_model"),reverse_direction=paste0("A_model -> ",feature),forward_cs=z$cs$y_xmap_x_mean,reverse_cs=z$cs$x_xmap_y_mean,forward_xmap=z$xmap$y_xmap_x_mean,reverse_xmap=z$xmap$x_xmap_y_mean,forward_sig=z$xmap$y_xmap_x_sig,reverse_sig=z$xmap$x_xmap_y_sig,forward_lower=z$xmap$y_xmap_x_lower,reverse_lower=z$xmap$x_xmap_y_lower,forward_upper=z$xmap$y_xmap_x_upper,reverse_upper=z$xmap$x_xmap_y_upper,neighbors=z$xmap$neighbors,libsizes=z$cs$libsizes,forward_minus_reverse=z$cs$y_xmap_x_mean-z$cs$x_xmap_y_mean,runtime_seconds=proc.time()[[3]]-t0,stringsAsFactors=FALSE)
}
if (mode=="full") {
  fnn_rows <- list(); q<-1
  for(s in scenarios) for(v in c(features,"A_model")) {
    val <- sf::st_drop_geometry(datasets[[s]])[[v]]
    f <- capture_warn(spEDM::fnn(datasets[[s]],target=v,E=1:10,tau=1,style=1,detrend=FALSE,eps=sd(val),threads=1,nb=nb))
    returned_E <- as.integer(sub("E:","",names(f),fixed=TRUE))
    stopifnot(length(f)>0,all(returned_E %in% 1:10))
    fnn_rows[[q]]<-data.frame(scenario=s,variable=v,requested_E_max=10,E=returned_E,fnn=as.numeric(f),eps=sd(val),truncated=(length(f)<10));q<-q+1
  }
  do.call(rbind,fnn_rows) |> write.csv(file.path(record,"fnn_curves.csv"),row.names=FALSE)
  rows<-list(); q<-1
  for(style in c(1,0)) for(E in c(3,6,8)) for(k in c(E^2,60)) for(s in scenarios) for(feature in features) { rows[[q]]<-run_one(s,feature,E,k,style);q<-q+1;cat("DONE",style,E,k,s,feature,"\n") }
  all<-do.call(rbind,rows)
  write.csv(all,file.path(record,"gcmc_direction_results.csv"),row.names=FALSE)
  write.csv(all,file.path(record,"gcmc_sensitivity_results.csv"),row.names=FALSE)
  write.csv(subset(all,style==1 & E==3 & k==9),file.path(record,"gcmc_primary_results.csv"),row.names=FALSE)
  writeLines(unique(warns),file.path(record,"r_warnings.txt"))
} else if(mode=="repro") {
  rows<-list();q<-1
  for(s in scenarios) for(feature in features){rows[[q]]<-run_one(s,feature,3,9,1);q<-q+1}
  all<-do.call(rbind,rows)
  write.csv(all,file.path(record,"gcmc_primary_repro_results.csv"),row.names=FALSE)
  write.csv(all,file.path(record,"gcmc_direction_repro_results.csv"),row.names=FALSE)
} else stop("unknown mode")
