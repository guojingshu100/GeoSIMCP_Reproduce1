
options(warn = 1)

suppressPackageStartupMessages({
    library(parallel)
    library(foreach)
    library(doParallel)
})

cat(
    "============================================================\n"
)
cat(
    "PM2.5 TRAIN-ONLY GCCM\n"
)
cat(
    "============================================================\n"
)

# ============================================================
# Paths
# ============================================================

input_path <- "E:/GeoSIMCP_Reproduce/geoconformal-main/GCCP_PM2.5/outputs/gccm/pm25_gccm_train_detrended_v1.csv"
edge_path  <- "E:/GeoSIMCP_Reproduce/geoconformal-main/GCCP_PM2.5/outputs/gccm/pm25_gccm_delaunay_edges_v1.csv"
source_path <- "E:/GeoSIMCP_Reproduce/geoconformal-main/GCCP_Seattle/02_gccm/GCCM_updated 2/R/GCCM/GCCM4Lattice.r"

raw_output  <- "E:/GeoSIMCP_Reproduce/geoconformal-main/GCCP_PM2.5/outputs/gccm/pm25_gccm_raw_rho_v1.csv"
mean_output <- "E:/GeoSIMCP_Reproduce/geoconformal-main/GCCP_PM2.5/outputs/gccm/pm25_gccm_mean_curves_v1.csv"

source(source_path)

cat("GCCM source loaded.\n")

# ============================================================
# Read Train-only data
# ============================================================

dat <- read.csv(
    input_path,
    check.names = FALSE,
    fileEncoding = "UTF-8-BOM"
)

edges <- read.csv(
    edge_path,
    check.names = FALSE,
    fileEncoding = "UTF-8-BOM"
)

n <- nrow(dat)

cat("Train sites:", n, "\n")

if (n != 990) {
    stop(
        paste0(
            "Expected 990 Train sites, got ",
            n
        )
    )
}

# ============================================================
# Build neighborMatrix
#
# GCCM uses 1-based integer site indices.
# ============================================================

neighborMatrix <- vector(
    "list",
    n
)

for (i in seq_len(n)) {
    neighborMatrix[[i]] <- integer(0)
}

for (i in seq_len(nrow(edges))) {

    u <- as.integer(
        edges$u1[i]
    )

    v <- as.integer(
        edges$v1[i]
    )

    neighborMatrix[[u]] <- c(
        neighborMatrix[[u]],
        v
    )

    neighborMatrix[[v]] <- c(
        neighborMatrix[[v]],
        u
    )
}

neighborMatrix <- lapply(
    neighborMatrix,
    function(x) {
        sort(
            unique(
                as.integer(x)
            )
        )
    }
)

degree <- lengths(
    neighborMatrix
)

cat(
    "Mean degree:",
    mean(degree),
    "\n"
)

cat(
    "Minimum degree:",
    min(degree),
    "\n"
)

if (any(degree == 0)) {
    stop(
        "At least one site has no Delaunay neighbor."
    )
}

# ============================================================
# Frozen GCCM settings
# ============================================================

E <- 3
tau <- 1
b <- 4

lib <- c(
    1,
    n
)

pred <- c(
    1,
    n
)

max_lib_expected <- (
    n
    - (E - 1) * tau
)

lib_sizes <- c(
    10,
    50,
    100,
    200,
    400,
    max_lib_expected
)

cat("\n")
cat(
    "E                =",
    E,
    "\n"
)

cat(
    "tau              =",
    tau,
    "\n"
)

cat(
    "b                =",
    b,
    "\n"
)

cat(
    "max library size =",
    max_lib_expected,
    "\n"
)

cat(
    "library sizes    =",
    paste(
        lib_sizes,
        collapse = ", "
    ),
    "\n"
)

if (max_lib_expected != 988) {
    stop(
        "Expected max library size = 988."
    )
}

# ============================================================
# Variables
# ============================================================

features <- c(
    "dem",
    "sp",
    "tp",
    "blh",
    "r",
    "aod_sat",
    "ndvi"
)

effect_name <- "PM2.5"

effect_col <- paste0(
    effect_name,
    "_detrended"
)

y <- as.numeric(
    dat[[effect_col]]
)

if (any(!is.finite(y))) {
    stop(
        "PM2.5 detrended target contains non-finite values."
    )
}

# ============================================================
# Helper: aggregate GCCM rho for each library size
#
# Match the original polygon workflow:
#
# max(0, mean(rho, na.rm=TRUE))
# ============================================================

mean_curve <- function(
    res,
    feature_name,
    direction_name
) {

    split_res <- split(
        res,
        res$L
    )

    rho_mean <- sapply(
        split_res,
        function(z) {

            temp <- mean(
                z$rho,
                na.rm = TRUE
            )

            max(
                0,
                temp
            )
        }
    )

    out <- data.frame(
        feature = feature_name,
        direction = direction_name,
        lib_size = as.integer(
            names(rho_mean)
        ),
        rho = as.numeric(
            rho_mean
        )
    )

    out <- out[
        order(
            out$lib_size
        ),
    ]

    rownames(out) <- NULL

    return(out)
}

# ============================================================
# Run
# ============================================================

all_raw <- list()
all_mean <- list()

raw_counter <- 1
mean_counter <- 1

# No random sampling is used in GCCMSingle4Lattice,
# but fixing seed makes the runner fully explicit.
set.seed(42)

for (feature_name in features) {

    cat("\n")
    cat(
        "------------------------------------------------------------\n"
    )

    cat(
        "Feature:",
        feature_name,
        "\n"
    )

    x_col <- paste0(
        feature_name,
        "_detrended"
    )

    x <- as.numeric(
        dat[[x_col]]
    )

    if (any(!is.finite(x))) {
        stop(
            paste(
                feature_name,
                "contains non-finite values"
            )
        )
    }

    # --------------------------------------------------------
    # Embedding from candidate cause X
    # --------------------------------------------------------

    embed_x <- generateEmbedings(
        neighborMatrix,
        x,
        E
    )

    # --------------------------------------------------------
    # Embedding from effect PM2.5
    # --------------------------------------------------------

    embed_y <- generateEmbedings(
        neighborMatrix,
        y,
        E
    )

    if (
        any(!is.finite(embed_x))
        ||
        any(!is.finite(embed_y))
    ) {
        stop(
            paste(
                "Non-finite embedding for",
                feature_name
            )
        )
    }

    # --------------------------------------------------------
    # Direction 1:
    # X manifold reconstructs PM2.5
    # --------------------------------------------------------

    cat(
        "  Running cause_xmap_effect ...\n"
    )

    x_xmap_y <- GCCMLattice(
        embed_x,
        y,
        lib_sizes,
        lib,
        pred,
        E,
        tau = tau,
        b = b,
        cores = 6
    )

    x_xmap_y$feature <- feature_name
    x_xmap_y$direction <- "cause_xmap_effect"

    all_raw[[raw_counter]] <- x_xmap_y

    raw_counter <- raw_counter + 1

    all_mean[[mean_counter]] <- mean_curve(
        x_xmap_y,
        feature_name,
        "cause_xmap_effect"
    )

    mean_counter <- mean_counter + 1

    # --------------------------------------------------------
    # Direction 2:
    # PM2.5 manifold reconstructs X
    #
    # THIS is the pre-specified direction used later
    # for X -> PM2.5 causal strength.
    # --------------------------------------------------------

    cat(
        "  Running effect_xmap_cause ...\n"
    )

    y_xmap_x <- GCCMLattice(
        embed_y,
        x,
        lib_sizes,
        lib,
        pred,
        E,
        tau = tau,
        b = b,
        cores = 6
    )

    y_xmap_x$feature <- feature_name
    y_xmap_x$direction <- "effect_xmap_cause"

    all_raw[[raw_counter]] <- y_xmap_x

    raw_counter <- raw_counter + 1

    all_mean[[mean_counter]] <- mean_curve(
        y_xmap_x,
        feature_name,
        "effect_xmap_cause"
    )

    mean_counter <- mean_counter + 1

    cat(
        "  Finished:",
        feature_name,
        "\n"
    )
}

# ============================================================
# Combine and save
# ============================================================

raw_df <- do.call(
    rbind,
    all_raw
)

mean_df <- do.call(
    rbind,
    all_mean
)

raw_df <- raw_df[
    ,
    c(
        "feature",
        "direction",
        "L",
        "rho"
    )
]

names(raw_df)[
    names(raw_df) == "L"
] <- "lib_size"

mean_df <- mean_df[
    order(
        mean_df$feature,
        mean_df$direction,
        mean_df$lib_size
    ),
]

rownames(raw_df) <- NULL
rownames(mean_df) <- NULL

write.csv(
    raw_df,
    raw_output,
    row.names = FALSE,
    fileEncoding = "UTF-8"
)

write.csv(
    mean_df,
    mean_output,
    row.names = FALSE,
    fileEncoding = "UTF-8"
)

cat("\n")
cat(
    "============================================================\n"
)

cat(
    "GCCM COMPLETE\n"
)

cat(
    "============================================================\n"
)

cat(
    "Raw rows :",
    nrow(raw_df),
    "\n"
)

cat(
    "Mean rows:",
    nrow(mean_df),
    "\n"
)

cat(
    "Raw output :",
    raw_output,
    "\n"
)

cat(
    "Mean output:",
    mean_output,
    "\n"
)
