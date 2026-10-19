args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 3) stop("usage: script RECORD STAGE10B_RECORD LOCAL_LIBRARY")

record <- normalizePath(args[[1]], winslash = "/", mustWork = TRUE)
stage10b <- normalizePath(args[[2]], winslash = "/", mustWork = TRUE)
local_lib <- normalizePath(args[[3]], winslash = "/", mustWork = TRUE)
.libPaths(c(local_lib, .libPaths()))
stopifnot(as.character(packageVersion("spEDM")) == "1.12")

features <- c("x1", "x2", "g_proxy", "x_noise")
scenarios <- c("S2", "S4")
edges <- read.csv(file.path(stage10b, "inputs", "delaunay_edges.csv"), check.names = FALSE)

make_nb <- function(n, edges) {
  nb <- lapply(seq_len(n), function(i) {
    sort(unique(c(edges$v[edges$u == i], edges$u[edges$v == i])))
  })
  nb <- lapply(nb, as.integer)
  class(nb) <- "nb"
  attr(nb, "region.id") <- as.character(seq_len(n))
  attr(nb, "sym") <- TRUE
  nb
}

nb <- make_nb(200, edges)
load_scientific_table <- function(scenario) {
  d <- read.csv(file.path(stage10b, "inputs", paste0(scenario, "_scale_fit_gcmc.csv")), check.names = FALSE)
  stopifnot(
    nrow(d) == 200,
    !anyDuplicated(d$sample_id),
    identical(names(d), c("sample_id", "s_x", "s_y", features, "A_model")),
    all(is.finite(as.matrix(d[c("s_x", "s_y", features, "A_model")]))))
  sf::st_as_sf(d, coords = c("s_x", "s_y"), remove = TRUE, crs = NA)
}
datasets <- setNames(lapply(scenarios, load_scientific_table), scenarios)

warnings_seen <- character()
with_warnings <- function(expr) {
  withCallingHandlers(expr, warning = function(w) {
    warnings_seen <<- c(warnings_seen, conditionMessage(w))
    invokeRestart("muffleWarning")
  })
}

run_one <- function(scenario, feature, k, style) {
  started <- proc.time()[[3]]
  g <- with_warnings(spEDM::gcmc(
    datasets[[scenario]], cause = feature, effect = "A_model",
    libsizes = 200, E = 3, k = k, tau = 1, style = style,
    dist.metric = "L2", threads = 1, detrend = FALSE,
    parallel.level = "low", bidirectional = TRUE,
    progressbar = FALSE, nb = nb
  ))
  stopifnot(
    identical(g$varname, c(feature, "A_model")),
    identical(names(g$xmap), c(
      "neighbors", "x_xmap_y_mean", "x_xmap_y_sig", "x_xmap_y_lower", "x_xmap_y_upper",
      "y_xmap_x_mean", "y_xmap_x_sig", "y_xmap_x_lower", "y_xmap_x_upper")),
    identical(names(g$cs), c("libsizes", "x_xmap_y_mean", "y_xmap_x_mean")),
    nrow(g$xmap) == 1,
    nrow(g$cs) == 1
  )
  data.frame(
    scenario = scenario,
    feature = feature,
    cause = g$varname[1],
    effect = g$varname[2],
    E = 3,
    k = k,
    tau = 1,
    style = style,
    xmap_neighbors = g$xmap$neighbors,
    xmap_x_xmap_y_mean = g$xmap$x_xmap_y_mean,
    xmap_x_xmap_y_sig = g$xmap$x_xmap_y_sig,
    xmap_x_xmap_y_lower = g$xmap$x_xmap_y_lower,
    xmap_x_xmap_y_upper = g$xmap$x_xmap_y_upper,
    xmap_y_xmap_x_mean = g$xmap$y_xmap_x_mean,
    xmap_y_xmap_x_sig = g$xmap$y_xmap_x_sig,
    xmap_y_xmap_x_lower = g$xmap$y_xmap_x_lower,
    xmap_y_xmap_x_upper = g$xmap$y_xmap_x_upper,
    cs_libsizes = g$cs$libsizes,
    cs_x_xmap_y_mean = g$cs$x_xmap_y_mean,
    cs_y_xmap_x_mean = g$cs$y_xmap_x_mean,
    xmap_forward_field = "y_xmap_x_mean",
    cs_forward_field = "y_xmap_x_mean",
    runtime_seconds = proc.time()[[3]] - started,
    stringsAsFactors = FALSE
  )
}

run_grid <- function(settings) {
  rows <- list()
  index <- 1L
  for (setting_index in seq_len(nrow(settings))) {
    for (scenario in scenarios) {
      for (feature in features) {
        rows[[index]] <- run_one(
          scenario, feature,
          settings$k[[setting_index]],
          settings$style[[setting_index]])
        index <- index + 1L
      }
    }
  }
  do.call(rbind, rows)
}

original <- run_grid(data.frame(k = 9L, style = 1L))
vignette_k <- run_grid(data.frame(k = c(25L, 25L), style = c(1L, 0L)))
write.csv(original, file.path(record, "gcmc_xmap_vs_cs_primary.csv"), row.names = FALSE)
write.csv(vignette_k, file.path(record, "gcmc_vignette_k25_results.csv"), row.names = FALSE)
writeLines(unique(warnings_seen), file.path(record, "r_warnings.txt"))

# Optional package-bundled example: no project data and no external downloads.
example_file <- system.file("case/columbus.gpkg", package = "spEDM")
if (nzchar(example_file) && requireNamespace("sf", quietly = TRUE) && requireNamespace("ggplot2", quietly = TRUE)) {
  columbus <- sf::read_sf(example_file)
  example_g <- with_warnings(spEDM::gcmc(
    columbus, "hoval", "crime", E = 7, k = 19,
    threads = 1, progressbar = FALSE
  ))
  writeLines(capture.output(print(example_g)), file.path(record, "official_example_print.txt"))
  write.csv(example_g$xmap, file.path(record, "official_example_xmap.csv"), row.names = FALSE)
  write.csv(example_g$cs, file.path(record, "official_example_cs.csv"), row.names = FALSE)
  example_plot <- plot(example_g)
  write.csv(example_plot$data, file.path(record, "official_example_plot_data.csv"), row.names = FALSE)
  writeLines(
    c(
      "status=PASS",
      paste0("class=", paste(class(example_g), collapse = ",")),
      paste0("names=", paste(names(example_g), collapse = ",")),
      paste0("plot_class=", paste(class(example_plot), collapse = ",")),
      paste0("print_source=xmap (first list element)"),
      paste0("plot_source=cs (x[-1], then first list element)")
    ),
    file.path(record, "official_example_status.txt")
  )
} else {
  writeLines("status=SKIPPED; bundled data or already-installed package unavailable", file.path(record, "official_example_status.txt"))
}

writeLines(unique(warnings_seen), file.path(record, "r_warnings.txt"))
cat("STAGE10B2_R_PASS\n")
