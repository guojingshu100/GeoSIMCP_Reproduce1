
library(parallel)
library(foreach)
library(doParallel)

source("E:/GeoSIMCP_Reproduce/geoconformal-main/GCCP_Seattle/02_gccm/GCCM_updated 2/R/GCCM/GCCM4Lattice.r")
source("E:/GeoSIMCP_Reproduce/geoconformal-main/GCCP_Seattle/02_gccm/GCCM_updated 2/R/GCCM/basic.r")

sites <- read.csv(
    "E:/GeoSIMCP_Reproduce/geoconformal-main/GCCP_Seattle/outputs/gccm/seattle_gccm_outer_train_sites.csv",
    check.names = FALSE
)

edges <- read.csv(
    "E:/GeoSIMCP_Reproduce/geoconformal-main/GCCP_Seattle/outputs/gccm/seattle_gccm_outer_train_delaunay_edges.csv"
)

n <- nrow(sites)


# ------------------------------------------------------------
# Delaunay adjacency
# ------------------------------------------------------------

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


# ------------------------------------------------------------
# Frozen GCCM parameters
# ------------------------------------------------------------

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


# ------------------------------------------------------------
# Scientific hypothesis:
#
# sqft_living -> price
#
# GCCM reverse map:
#
# price xmap sqft_living
#
# ------------------------------------------------------------

x <- as.numeric(
    sites$price_resid
)

y <- as.numeric(
    sites$sqft_living_resid
)


# Price manifold
embeddings <- generateEmbedings(
    neighborMatrix,
    x,
    E
)


cat(
    "\nStarting full GCCM run...\n"
)

start_time <- Sys.time()


# ------------------------------------------------------------
# Use official GCCMLattice unchanged.
#
# cores = 8:
# official polygon example also uses 8 cores.
# No random sampling is introduced by this function.
# ------------------------------------------------------------

result <- GCCMLattice(
    embeddings,
    y,
    lib_sizes,
    lib,
    pred,
    E,
    b = b,
    cores = 8
)


end_time <- Sys.time()


write.csv(
    result,
    "E:/GeoSIMCP_Reproduce/geoconformal-main/GCCP_Seattle/outputs/gccm_author_aligned/smoke_test/old_price_xmap_sqft_living_full_reproduced.csv",
    row.names = FALSE
)


cat(
    "\n===== FULL GCCM RUN FINISHED =====\n"
)

cat(
    "Runtime minutes:",
    as.numeric(
        difftime(
            end_time,
            start_time,
            units = "mins"
        )
    ),
    "\n"
)

cat(
    "Rows:",
    nrow(result),
    "\n"
)

print(
    table(
        result$L
    )
)

cat(
    "\nMean rho by L:\n"
)

print(
    aggregate(
        rho ~ L,
        data = result,
        FUN = mean,
        na.rm = TRUE
    )
)
