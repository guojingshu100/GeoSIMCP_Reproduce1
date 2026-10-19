args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 5) stop("usage: script INPUT EDGES OUTPUT MECHANISM LOCAL_LIBRARY")
input_path <- normalizePath(args[[1]], winslash = "/", mustWork = TRUE)
edge_path <- normalizePath(args[[2]], winslash = "/", mustWork = TRUE)
output_path <- args[[3]]
mechanism_requested <- args[[4]]
local_lib <- normalizePath(args[[5]], winslash = "/", mustWork = TRUE)
.libPaths(c(local_lib, .libPaths()))
stopifnot(as.character(packageVersion("spEDM")) == "1.12")

features <- c("x1", "x2", "g_proxy", "x_noise", "Z")
mechanisms <- c("C0_INDEPENDENT_NULL", "C1_RESIDUAL_SCALE_CAUSAL", "C2_PURE_CONFOUNDING", "C3_REVERSE_DIRECTION", "C4_SPATIAL_LAG_CAUSAL", "C5_NONLINEAR_DIRECT")
mechanism_index <- match(mechanism_requested, mechanisms)
if (is.na(mechanism_index)) stop("unknown mechanism")
edges <- read.csv(edge_path, check.names = FALSE)
make_nb <- function(n, edges) {
  nb <- lapply(seq_len(n), function(i) sort(unique(c(edges$v[edges$u == i], edges$u[edges$v == i]))))
  nb <- lapply(nb, as.integer)
  class(nb) <- "nb"
  attr(nb, "region.id") <- as.character(seq_len(n))
  attr(nb, "sym") <- TRUE
  nb
}
nb <- make_nb(200, edges)
d <- read.csv(input_path, check.names = FALSE)
d <- d[d$mechanism == mechanism_requested, , drop = FALSE]
stopifnot(nrow(d) == 50L * 200L, !anyDuplicated(d[c("dataset_id", "sample_id")]))

warnings_seen <- character()
capture_warning <- function(expr) withCallingHandlers(expr, warning = function(w) {
  warnings_seen <<- c(warnings_seen, conditionMessage(w)); invokeRestart("muffleWarning")
})
rows <- list()
out_index <- 1L
groups <- split(d, d$dataset_id, drop = TRUE)
for (dataset_id in sort(names(groups))) {
  group <- groups[[dataset_id]]
  group <- group[order(group$row_index), , drop = FALSE]
  rep_index <- as.integer(group$replicate_index[[1]])
  base_cols <- c(features, "A", "s_x", "s_y")
  scientific <- sf::st_as_sf(group[base_cols], coords = c("s_x", "s_y"), remove = TRUE, crs = NA)
  for (feature_index in seq_along(features)) {
    feature <- features[[feature_index]]
    conds <- setdiff(features, feature)
    common <- data.frame(
      dataset_id = dataset_id, mechanism = mechanism_requested,
      replicate_index = rep_index, generation_seed = group$generation_seed[[1]],
      feature = feature, stringsAsFactors = FALSE
    )
    started <- proc.time()[[3]]
    g <- capture_warning(spEDM::gpc(
      scientific, cause = feature, effect = "A", E = 3, k = 5, tau = 1, style = 1,
      dist.metric = "L2", detrend = TRUE, bidirectional = TRUE, threads = 1,
      progressbar = FALSE, nb = nb
    ))
    gs <- g$summary
    forward_gpc <- sum(gs$strength[gs$direction == "y_xmap_x"])
    reverse_gpc <- sum(gs$strength[gs$direction == "x_xmap_y"])
    g_types <- setNames(gs$strength[gs$direction == "y_xmap_x"], gs$type[gs$direction == "y_xmap_x"])
    gr_types <- setNames(gs$strength[gs$direction == "x_xmap_y"], gs$type[gs$direction == "x_xmap_y"])
    rows[[out_index]] <- cbind(common, data.frame(
      method = "GPC", forward_score = forward_gpc, reverse_score = reverse_gpc,
      forward_p = NA_real_, reverse_p = NA_real_,
      forward_positive = unname(g_types[["positive"]]), forward_negative = unname(g_types[["negative"]]), forward_dark = unname(g_types[["dark"]]),
      reverse_positive = unname(gr_types[["positive"]]), reverse_negative = unname(gr_types[["negative"]]), reverse_dark = unname(gr_types[["dark"]]),
      pxmap_forward = NA_real_, pxmap_reverse = NA_real_, xmap_forward = NA_real_, xmap_reverse = NA_real_,
      runtime_seconds = proc.time()[[3]] - started, stringsAsFactors = FALSE
    )); out_index <- out_index + 1L

    started <- proc.time()[[3]]
    sct_seed <- 20600000L + (mechanism_index - 1L) * 100000L + rep_index * 100L + feature_index
    s <- capture_warning(spEDM::sc.test(
      scientific, cause = feature, effect = "A", k = 15, block = 3, boot = 399,
      seed = sct_seed, base = 2, threads = 1, detrend = TRUE, normalize = FALSE, nb = nb
    ))
    stopifnot(length(s$sc) == 4L)
    rows[[out_index]] <- cbind(common, data.frame(
      method = "SCT", forward_score = unname(s$sc[[1]]), reverse_score = unname(s$sc[[3]]),
      forward_p = unname(s$sc[[2]]), reverse_p = unname(s$sc[[4]]),
      forward_positive = NA_real_, forward_negative = NA_real_, forward_dark = NA_real_,
      reverse_positive = NA_real_, reverse_negative = NA_real_, reverse_dark = NA_real_,
      pxmap_forward = NA_real_, pxmap_reverse = NA_real_, xmap_forward = NA_real_, xmap_reverse = NA_real_,
      runtime_seconds = proc.time()[[3]] - started, stringsAsFactors = FALSE
    )); out_index <- out_index + 1L

    started <- proc.time()[[3]]
    p <- capture_warning(spEDM::scpcm(
      scientific, cause = feature, effect = "A", conds = conds, E = 3, k = 5,
      tau = 1, style = 1, stack = FALSE, dist.metric = "L2", dist.average = TRUE,
      theta = 1, algorithm = "simplex", threads = 1, detrend = TRUE,
      bidirectional = TRUE, cumulate = FALSE, nb = nb
    ))
    stopifnot(nrow(p$pxmap) == 1L, nrow(p$xmap) == 1L)
    rows[[out_index]] <- cbind(common, data.frame(
      method = "SCPCM", forward_score = p$pxmap$y_xmap_x_mean, reverse_score = p$pxmap$x_xmap_y_mean,
      forward_p = p$pxmap$y_xmap_x_sig, reverse_p = p$pxmap$x_xmap_y_sig,
      forward_positive = NA_real_, forward_negative = NA_real_, forward_dark = NA_real_,
      reverse_positive = NA_real_, reverse_negative = NA_real_, reverse_dark = NA_real_,
      pxmap_forward = p$pxmap$y_xmap_x_mean, pxmap_reverse = p$pxmap$x_xmap_y_mean,
      xmap_forward = p$xmap$y_xmap_x_mean, xmap_reverse = p$xmap$x_xmap_y_mean,
      runtime_seconds = proc.time()[[3]] - started, stringsAsFactors = FALSE
    )); out_index <- out_index + 1L
  }
  if (rep_index %% 10L == 0L) cat(mechanism_requested, "completed", rep_index, "/50\n")
}
out <- do.call(rbind, rows)
stopifnot(nrow(out) == 50L * 5L * 3L)
write.csv(out, output_path, row.names = FALSE)
writeLines(unique(warnings_seen), paste0(output_path, ".warnings.txt"))
cat("PASS", mechanism_requested, "rows", nrow(out), "\n")
