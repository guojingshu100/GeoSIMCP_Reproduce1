args <- commandArgs(trailingOnly = TRUE)
stopifnot(length(args) == 5)
input_path <- args[[1]]
edge_path <- args[[2]]
core_path <- args[[3]]
out_dir <- args[[4]]
mode <- args[[5]]
stopifnot(identical(mode, "D1"))

source(core_path, local = .GlobalEnv)
dat <- read.csv(input_path, check.names = FALSE, stringsAsFactors = FALSE)
features <- c("s_x", "s_y", "x1", "x2", "g_proxy", "x_noise")
expected <- c("node_id", "sample_id", "s_x", "s_y", "abs_residual", paste0("z_", features))
stopifnot(identical(names(dat), expected), nrow(dat) == 200L)
stopifnot(identical(dat$node_id, seq_len(200L)), identical(dat$sample_id, sort(dat$sample_id)))
edges <- read.csv(edge_path, stringsAsFactors = FALSE)
stopifnot(identical(names(edges), c("u", "v")))
nb <- lapply(seq_len(200L), function(i) sort(unique(c(edges$v[edges$u == i], edges$u[edges$v == i]))))
stopifnot(all(lengths(nb) > 0))

# The common embedding is deliberately made once.  D1 changes only lib/pred membership.
E <- 3L; tau <- 1L; b <- 4L
libs_common <- c(10L, 18L, 33L, 60L, 109L, 198L)
emb <- generateEmbedings(nb, dat$abs_residual, E)
stopifnot(identical(dim(emb), c(200L, 3L)), all(is.finite(emb)))

full_single <- function(target, lib_size) {
  GCCMSingle4Lattice(
    emb, target, rep.int(TRUE, 200L), lib_size, 200L, seq_len(200L),
    rep.int(TRUE, 200L), b
  )
}

raw_a <- list(); raw_b <- list(); raw_b200 <- list()
for (feature in features) {
  target <- dat[[paste0("z_", feature)]]
  # Condition A is the unmodified historical public entry point.
  a <- GCCMLattice(emb, target, libs_common, c(1L, 200L), c(1L, 200L), E, tau = tau, b = b, cores = NULL)
  a$feature <- feature; a$direction <- "effect_xmap_cause"; a$condition <- "historical_198"; a$comparison_role <- "paired_common"
  raw_a[[feature]] <- a
  # Condition B retains the same historical single-library routine and only supplies all 200 spatial IDs.
  braw <- do.call(rbind, lapply(libs_common, function(L) full_single(target, L)))
  braw$feature <- feature; braw$direction <- "effect_xmap_cause"; braw$condition <- "full_200"; braw$comparison_role <- "paired_common"
  raw_b[[feature]] <- braw
  bsup <- full_single(target, 200L)
  bsup$feature <- feature; bsup$direction <- "effect_xmap_cause"; bsup$condition <- "full_200"; bsup$comparison_role <- "supplementary_L200"
  raw_b200[[feature]] <- bsup
}

write.csv(do.call(rbind, raw_a), file.path(out_dir, "raw_historical_198.csv"), row.names = FALSE, na = "")
write.csv(do.call(rbind, raw_b), file.path(out_dir, "raw_full_200_paired.csv"), row.names = FALSE, na = "")
write.csv(do.call(rbind, raw_b200), file.path(out_dir, "raw_full_200_supplementary_L200.csv"), row.names = FALSE, na = "")
write.csv(data.frame(node_id = dat$node_id, sample_id = dat$sample_id, embedding_1 = emb[,1], embedding_2 = emb[,2], embedding_3 = emb[,3]), file.path(out_dir, "shared_embedding.csv"), row.names = FALSE)
write.csv(data.frame(key = c("n_original", "n_historical_lib_pred", "n_full_lib_pred", "E", "tau", "b", "common_libraries", "supplementary_library"), value = c(200, 198, 200, E, tau, b, paste(libs_common, collapse = ";"), 200)), file.path(out_dir, "adapter_metadata.csv"), row.names = FALSE)
