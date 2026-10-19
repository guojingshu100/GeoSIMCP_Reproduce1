
library(parallel)
library(foreach)
library(doParallel)

# ============================================================
# OFFICIAL GCCM CORE
#
# These source files are used unchanged.
# ============================================================

source("E:/GeoSIMCP_Reproduce/geoconformal-main/GCCP_Seattle/02_gccm/GCCM_updated 2/R/GCCM/GCCM4Lattice.r")
source("E:/GeoSIMCP_Reproduce/geoconformal-main/GCCP_Seattle/02_gccm/GCCM_updated 2/R/GCCM/basic.r")


# ============================================================
# SEATTLE AUTHOR-ALIGNED INPUT
# ============================================================

sites <- read.csv(
    "E:/GeoSIMCP_Reproduce/geoconformal-main/GCCP_Seattle/outputs/gccm_author_aligned/seattle_gccm_train_logprice_sites_v1.csv",
    check.names = FALSE
)

edges <- read.csv(
    "E:/GeoSIMCP_Reproduce/geoconformal-main/GCCP_Seattle/outputs/gccm_author_aligned/seattle_gccm_train_delaunay_edges_v1.csv"
)

n <- nrow(sites)


cat(
    "\n============================================\n"
)

cat(
    "Seattle author-aligned log_price GCCM\n"
)

cat(
    "============================================\n"
)

cat(
    "Sites:",
    n,
    "\n"
)


stopifnot(
    n == 2088
)


# ============================================================
# SEATTLE POINT-GRAPH ADAPTATION
#
# Reconstruct Delaunay adjacency list.
# ============================================================

neighborMatrix <- vector(
    "list",
    n
)


for (i in 1:n) {

    neighborMatrix[[i]] <- as.integer(
        edges$to_site[
            edges$from_site == i
        ]
    )
}


stopifnot(
    length(neighborMatrix) == n
)


stopifnot(
    all(
        sapply(
            neighborMatrix,
            length
        ) > 0
    )
)


# ============================================================
# FROZEN GCCM SETTINGS
# ============================================================

E <- 3

b <- E + 1

lib <- c(
    1,
    n
)

pred <- c(
    1,
    n
)

lib_sizes <- c(
    10,
    50,
    100,
    200,
    400,
    2086
)


cat(
    "E:",
    E,
    "\n"
)

cat(
    "b:",
    b,
    "\n"
)

cat(
    "Library sizes:",
    paste(
        lib_sizes,
        collapse = ", "
    ),
    "\n"
)


# ============================================================
# CAUSAL DIRECTION
#
# Scientific hypothesis:
#
#     feature -> log_price
#
# Under CCM/GCCM reverse mapping:
#
#     log_price xmap feature
#
# Therefore:
#
# embedding/manifold variable = log_price_resid
# reconstruction target       = feature_resid
# ============================================================

x_effect <- as.numeric(
    sites$log_price_resid
)


# Build log_price spatial manifold ONCE.
#
# It is identical for all 8 feature targets.
effect_embeddings <- generateEmbedings(
    neighborMatrix,
    x_effect,
    E
)


features <- c("bathrooms","sqft_living","sqft_lot","grade","condition","waterfront","view","age")

output_paths <- c("bathrooms"="E:/GeoSIMCP_Reproduce/geoconformal-main/GCCP_Seattle/outputs/gccm_author_aligned/seattle_gccm_train_logprice_xmap_bathrooms_convergence_raw_v1.csv","sqft_living"="E:/GeoSIMCP_Reproduce/geoconformal-main/GCCP_Seattle/outputs/gccm_author_aligned/seattle_gccm_train_logprice_xmap_sqft_living_convergence_raw_v1.csv","sqft_lot"="E:/GeoSIMCP_Reproduce/geoconformal-main/GCCP_Seattle/outputs/gccm_author_aligned/seattle_gccm_train_logprice_xmap_sqft_lot_convergence_raw_v1.csv","grade"="E:/GeoSIMCP_Reproduce/geoconformal-main/GCCP_Seattle/outputs/gccm_author_aligned/seattle_gccm_train_logprice_xmap_grade_convergence_raw_v1.csv","condition"="E:/GeoSIMCP_Reproduce/geoconformal-main/GCCP_Seattle/outputs/gccm_author_aligned/seattle_gccm_train_logprice_xmap_condition_convergence_raw_v1.csv","waterfront"="E:/GeoSIMCP_Reproduce/geoconformal-main/GCCP_Seattle/outputs/gccm_author_aligned/seattle_gccm_train_logprice_xmap_waterfront_convergence_raw_v1.csv","view"="E:/GeoSIMCP_Reproduce/geoconformal-main/GCCP_Seattle/outputs/gccm_author_aligned/seattle_gccm_train_logprice_xmap_view_convergence_raw_v1.csv","age"="E:/GeoSIMCP_Reproduce/geoconformal-main/GCCP_Seattle/outputs/gccm_author_aligned/seattle_gccm_train_logprice_xmap_age_convergence_raw_v1.csv")


# ============================================================
# RUN 8 FEATURES SEQUENTIALLY
#
# Each GCCMLattice call internally uses 8 cores.
# ============================================================

runtime_summary <- data.frame(
    feature = character(),
    runtime_minutes = numeric(),
    rows = integer(),
    rho_Lmax = numeric(),
    stringsAsFactors = FALSE
)


for (
    feature_index
    in seq_along(
        features
    )
) {

    feature <- features[
        feature_index
    ]


    target_col <- paste0(
        feature,
        "_resid"
    )


    cat(
        "\n--------------------------------------------\n"
    )

    cat(
        "[",
        feature_index,
        "/",
        length(features),
        "] ",
        feature,
        "\n",
        sep = ""
    )

    cat(
        "Direction: log_price xmap ",
        feature,
        "\n",
        sep = ""
    )


    y_feature <- as.numeric(
        sites[[target_col]]
    )


    start_time <- Sys.time()


    result <- GCCMLattice(

        effect_embeddings,

        y_feature,

        lib_sizes,

        lib,

        pred,

        E,

        b = b,

        cores = 8
    )


    end_time <- Sys.time()


    runtime_minutes <- as.numeric(
        difftime(
            end_time,
            start_time,
            units = "mins"
        )
    )


    write.csv(

        result,

        output_paths[
            feature
        ],

        row.names = FALSE
    )


    rho_Lmax <- result$rho[
        result$L == 2086
    ][1]


    runtime_summary <- rbind(

        runtime_summary,

        data.frame(

            feature =
                feature,

            runtime_minutes =
                runtime_minutes,

            rows =
                nrow(result),

            rho_Lmax =
                rho_Lmax,

            stringsAsFactors = FALSE
        )
    )


    cat(
        "Runtime minutes:",
        runtime_minutes,
        "\n"
    )

    cat(
        "Rows:",
        nrow(result),
        "\n"
    )

    cat(
        "rho(L=2086):",
        rho_Lmax,
        "\n"
    )


    cat(
        "Mean rho by L:\n"
    )

    print(
        aggregate(
            rho ~ L,
            data = result,
            FUN = mean,
            na.rm = TRUE
        )
    )
}


# ============================================================
# SAVE RUNTIME SUMMARY
# ============================================================

runtime_path <- file.path(

    dirname(
        output_paths[
            features[1]
        ]
    ),

    "seattle_gccm_train_logprice_runtime_summary_v1.csv"
)


write.csv(
    runtime_summary,
    runtime_path,
    row.names = FALSE
)


cat(
    "\n============================================\n"
)

cat(
    "ALL 8 GCCM RUNS FINISHED\n"
)

cat(
    "============================================\n"
)

print(
    runtime_summary
)

cat(
    "\nTotal runtime minutes:",
    sum(
        runtime_summary$runtime_minutes
    ),
    "\n"
)
