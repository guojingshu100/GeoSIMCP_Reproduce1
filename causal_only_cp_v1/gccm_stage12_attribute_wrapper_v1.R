args <- commandArgs(trailingOnly=TRUE)
stopifnot(length(args)==6)
input <- args[1]; edgefile <- args[2]; core <- args[3]
paramfile <- args[4]; out <- args[5]; mode <- args[6]
stopifnot(mode %in% c("preflight","estimate"))
library(foreach); library(doParallel)
source(core, local=.GlobalEnv)
p <- read.csv(paramfile, stringsAsFactors=FALSE)
stopifnot(identical(names(p),c("name","value")), !anyDuplicated(p$name))
getp <- function(key) p$value[match(key,p$name)]
E <- as.integer(getp("E")); tau <- as.integer(getp("tau")); b <- as.integer(getp("b"))
stopifnot(E==3,tau==1,b==4,getp("direction")=="effect_xmap_cause")
set.seed(as.integer(getp("seed")))

# Stage 12 changes only the candidate list required by the causal-only
# ablation.  Every per-feature GCCM calculation and validity gate below is
# identical to the frozen point-graph wrapper v1.
features <- c("x1","x2","g_proxy","x_noise")
dat <- read.csv(input,check.names=FALSE,stringsAsFactors=FALSE)
expected <- c("node_id","sample_id","s_x","s_y","abs_residual",paste0("z_",features))
stopifnot(identical(names(dat),expected),nrow(dat)==200L)
n <- nrow(dat)
stopifnot(identical(dat$node_id,seq_len(n)),!anyDuplicated(dat$sample_id),identical(dat$sample_id,sort(dat$sample_id)))
stopifnot(all(is.finite(as.matrix(dat[setdiff(expected,"sample_id")]))),all(dat$abs_residual>=0),sd(dat$abs_residual)>0)
stopifnot(all(vapply(dat[paste0("z_",features)],sd,numeric(1))>0))
edges <- read.csv(edgefile)
stopifnot(identical(names(edges),c("u","v")),all(edges$u>=1),all(edges$v<=n),all(edges$u<edges$v),!anyDuplicated(edges))
nb <- lapply(seq_len(n),function(i) sort(unique(c(edges$v[edges$u==i],edges$u[edges$v==i]))))
stopifnot(all(lengths(nb)>0))
rings <- sapply(1:E,function(k) lengths(laggedVar4Lattic(nb,k)))
stopifnot(all(rings>0))
emb <- generateEmbedings(nb,dat$abs_residual,E)
stopifnot(identical(dim(emb),c(n,E)),all(is.finite(emb)))
M <- n-(E-1)*tau
write.csv(data.frame(node_id=dat$node_id,sample_id=dat$sample_id,ring1=rings[,1],ring2=rings[,2],ring3=rings[,3]),file.path(out,"ring_counts.csv"),row.names=FALSE)
write.csv(data.frame(node_id=dat$node_id,sample_id=dat$sample_id,emb),file.path(out,"embedding.csv"),row.names=FALSE)
write.csv(data.frame(valid_embeddings=n,max_library=M,direction="effect_xmap_cause"),file.path(out,"preflight.csv"),row.names=FALSE)
capture.output(print(R.version),print(R.home()),print(.libPaths()),print(Sys.getlocale()),print(sessionInfo()),file=file.path(out,"sessionInfo.txt"))
if(mode=="preflight") quit(status=0)
L <- as.integer(strsplit(getp("library_sizes"),";",fixed=TRUE)[[1]])
stopifnot(length(L)==6,all(diff(L)>0),min(L)>=b+1,max(L)==M)
historical_compute_stats <- compute_stats
compute_stats <- function(obs,pred) {
  stopifnot(length(obs)==length(pred),all(is.finite(obs)),all(is.finite(pred)))
  result <- historical_compute_stats(obs,pred)
  stopifnot(is.finite(result$rho))
  result
}
raws <- list(); means <- list(); diags <- list()
for(feature in features) {
  raw <- GCCMLattice(emb,dat[[paste0("z_",feature)]],L,c(1,n),c(1,n),E,tau=tau,b=b,cores=NULL)
  stopifnot(nrow(raw)==(length(L)-1)*M+1,all(is.finite(raw$rho)))
  raw$feature <- feature; raw$direction <- "effect_xmap_cause"
  raw$window_index <- ave(seq_len(nrow(raw)),raw$L,FUN=seq_along)
  raws[[feature]] <- raw
  curve <- data.frame(L=L,rho=vapply(L,function(l) max(0,mean(raw$rho[raw$L==l])),numeric(1)))
  curve$feature <- feature; curve$direction <- "effect_xmap_cause"
  means[[feature]] <- curve
  terminal <- tail(curve$rho,1); gain <- terminal-curve$rho[1]
  mono <- if(length(unique(curve$rho))==1) NA_real_ else cor(curve$L,curve$rho,method="spearman")
  clipped <- min(1-1e-12,max(-1+1e-12,terminal))
  tstat <- clipped*sqrt((M-2)/(1-clipped^2))
  pval <- 2*pt(-abs(tstat),df=M-2)
  ci <- tanh(atanh(clipped)+c(-1,1)*qnorm(.975)/sqrt(M-3))
  converged <- is.finite(mono) && gain>0 && mono>=.60 && pval<.05 && ci[1]>0
  diags[[feature]] <- data.frame(feature=feature,direction="effect_xmap_cause",n_terminal=M,rho_Lmin=curve$rho[1],rho_Lmax=terminal,gain=gain,spearman=mono,p_two_sided=pval,CI_lower=ci[1],CI_upper=ci[2],converged=converged,strength=if(converged) terminal else 0)
}
raw <- do.call(rbind,raws); curve <- do.call(rbind,means); diag <- do.call(rbind,diags)
total <- sum(diag$strength)
weights <- data.frame(feature=diag$feature,strength=diag$strength,weight=if(total>0) diag$strength/total else rep(NA_real_,length(features)),converged=diag$converged)
# gamma is retained as historical metadata (the original six-variable wrapper
# used gamma=6*w); Stage 12 localization uses normalized weight, not gamma.
weights$gamma <- 6*weights$weight
weights$status <- if(total>0) "WEIGHTS_AVAILABLE" else "NO_CONVERGED_FEATURES"
write.csv(raw,file.path(out,"raw_curves.csv"),row.names=FALSE,na="")
write.csv(curve,file.path(out,"mean_curves.csv"),row.names=FALSE,na="")
write.csv(diag,file.path(out,"convergence.csv"),row.names=FALSE,na="")
write.csv(weights,file.path(out,"weights.csv"),row.names=FALSE,na="")
cat("STATUS:",if(total>0) "WEIGHTS_AVAILABLE" else "NO_CONVERGED_FEATURES","\n")
