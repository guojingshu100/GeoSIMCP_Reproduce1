args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 4) stop("usage: script RECORD STAGE10B_RECORD LOCAL_LIBRARY MODE")

record <- normalizePath(args[[1]], winslash = "/", mustWork = TRUE)
stage10b <- normalizePath(args[[2]], winslash = "/", mustWork = TRUE)
local_lib <- normalizePath(args[[3]], winslash = "/", mustWork = TRUE)
mode <- args[[4]]
.libPaths(c(local_lib, .libPaths()))
stopifnot(as.character(packageVersion("spEDM")) == "1.12")

edges <- read.csv(file.path(stage10b, "inputs", "delaunay_edges.csv"), check.names = FALSE)
make_nb <- function(n, edges) {
  nb <- lapply(seq_len(n), function(i) sort(unique(c(edges$v[edges$u == i], edges$u[edges$v == i]))))
  nb <- lapply(nb, as.integer)
  class(nb) <- "nb"
  attr(nb, "region.id") <- as.character(seq_len(n))
  attr(nb, "sym") <- TRUE
  nb
}
nb <- make_nb(200, edges)

warnings_seen <- character()
capture_warning <- function(expr) {
  withCallingHandlers(expr, warning = function(w) {
    warnings_seen <<- c(warnings_seen, conditionMessage(w))
    invokeRestart("muffleWarning")
  })
}

run_gcmc <- function(data_sf, style) {
  started <- proc.time()[[3]]
  g <- capture_warning(spEDM::gcmc(
    data_sf, cause = "X", effect = "A", libsizes = 200,
    E = 3, k = 25, tau = 1, style = style,
    dist.metric = "L2", detrend = FALSE, bidirectional = TRUE,
    threads = 1, parallel.level = "low", progressbar = FALSE, nb = nb
  ))
  stopifnot(
    identical(g$varname, c("X", "A")),
    nrow(g$xmap) == 1,
    nrow(g$cs) == 1
  )
  data.frame(
    forward_xmap = g$xmap$y_xmap_x_mean,
    reverse_xmap = g$xmap$x_xmap_y_mean,
    forward_sig = g$xmap$y_xmap_x_sig,
    reverse_sig = g$xmap$x_xmap_y_sig,
    forward_lower = g$xmap$y_xmap_x_lower,
    reverse_lower = g$xmap$x_xmap_y_lower,
    forward_upper = g$xmap$y_xmap_x_upper,
    reverse_upper = g$xmap$x_xmap_y_upper,
    xmap_neighbors = g$xmap$neighbors,
    forward_cs = g$cs$y_xmap_x_mean,
    reverse_cs = g$cs$x_xmap_y_mean,
    cs_libsizes = g$cs$libsizes,
    forward_minus_reverse = g$xmap$y_xmap_x_mean - g$xmap$x_xmap_y_mean,
    runtime_seconds = proc.time()[[3]] - started
  )
}

if (mode == "controlled") {
  d <- read.csv(file.path(record, "controlled_mechanisms.csv"), check.names = FALSE)
  stopifnot(nrow(d) == 50 * 5 * 200, !anyDuplicated(d[c("dataset_id", "sample_id")]))
  groups <- split(d, d$dataset_id, drop = TRUE)
  rows <- vector("list", length(groups) * 2)
  output_index <- 1L
  call_index <- 0L
  for (dataset_id in sort(names(groups))) {
    group <- groups[[dataset_id]]
    stopifnot(nrow(group) == 200, length(unique(group$mechanism)) == 1)
    scientific <- sf::st_as_sf(group[c("X", "A", "s_x", "s_y")], coords = c("s_x", "s_y"), remove = TRUE, crs = NA)
    for (style in c(1L, 0L)) {
      result <- run_gcmc(scientific, style)
      rows[[output_index]] <- cbind(
        data.frame(
          dataset_id = dataset_id,
          mechanism = group$mechanism[[1]],
          replicate_index = group$replicate_index[[1]],
          generation_seed = group$generation_seed[[1]],
          style = style,
          E = 3L,
          k = 25L,
          tau = 1L,
          stringsAsFactors = FALSE
        ),
        result
      )
      output_index <- output_index + 1L
      call_index <- call_index + 1L
      if (call_index %% 50L == 0L) cat("CONTROLLED_CALLS_COMPLETED", call_index, "\n")
    }
  }
  out <- do.call(rbind, rows)
  stopifnot(nrow(out) == 500)
  write.csv(out, file.path(record, "gcmc_controlled_results.csv"), row.names = FALSE)
  writeLines(unique(warnings_seen), file.path(record, "controlled_r_warnings.txt"))
  cat("STAGE10B3_CONTROLLED_PASS\n")
} else if (mode == "permutation") {
  scenarios <- c("S2", "S4")
  features <- c("x1", "x2", "g_proxy", "x_noise")
  rows <- vector("list", length(scenarios) * length(features) * 100L)
  output_index <- 1L
  call_index <- 0L
  for (scenario_index in seq_along(scenarios)) {
    scenario <- scenarios[[scenario_index]]
    d <- read.csv(file.path(stage10b, "inputs", paste0(scenario, "_scale_fit_gcmc.csv")), check.names = FALSE)
    stopifnot(nrow(d) == 200, !anyDuplicated(d$sample_id))
    for (feature_index in seq_along(features)) {
      feature <- features[[feature_index]]
      base <- data.frame(X = d[[feature]], A = d$A_model, s_x = d$s_x, s_y = d$s_y)
      observed_sf <- sf::st_as_sf(base, coords = c("s_x", "s_y"), remove = TRUE, crs = NA)
      observed <- run_gcmc(observed_sf, 1L)
      rows[[output_index]] <- cbind(
        data.frame(
          scenario = scenario, feature = feature, kind = "observed",
          permutation_index = 0L, permutation_seed = NA_integer_,
          style = 1L, E = 3L, k = 25L, tau = 1L,
          stringsAsFactors = FALSE
        ),
        observed
      )
      output_index <- output_index + 1L
      call_index <- call_index + 1L
      for (permutation_index in seq_len(99L)) {
        permutation_seed <- 20400000L + (scenario_index - 1L) * 100000L +
          (feature_index - 1L) * 1000L + permutation_index
        set.seed(permutation_seed)
        permuted <- base
        permuted$A <- sample(base$A, size = nrow(base), replace = FALSE)
        permuted_sf <- sf::st_as_sf(permuted, coords = c("s_x", "s_y"), remove = TRUE, crs = NA)
        result <- run_gcmc(permuted_sf, 1L)
        rows[[output_index]] <- cbind(
          data.frame(
            scenario = scenario, feature = feature, kind = "permuted",
            permutation_index = permutation_index, permutation_seed = permutation_seed,
            style = 1L, E = 3L, k = 25L, tau = 1L,
            stringsAsFactors = FALSE
          ),
          result
        )
        output_index <- output_index + 1L
        call_index <- call_index + 1L
        if (call_index %% 100L == 0L) cat("PERMUTATION_CALLS_COMPLETED", call_index, "\n")
      }
    }
  }
  out <- do.call(rbind, rows)
  stopifnot(nrow(out) == 800)
  write.csv(out, file.path(record, "observed_naive_permutation_results.csv"), row.names = FALSE)
  writeLines(unique(warnings_seen), file.path(record, "permutation_r_warnings.txt"))
  cat("STAGE10B3_PERMUTATION_PASS\n")
} else {
  stop("unknown mode")
}
