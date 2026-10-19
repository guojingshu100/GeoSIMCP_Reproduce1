
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
# Reconstruct adjacency list from directed edge table
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

# Basic topology checks
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
# Frozen GCCM settings
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

# GCCMLattice internally removes E-1 initial positions,
# giving max library = 2088 - 2 = 2086.
lib_sizes <- c(
    2086
)

# ------------------------------------------------------------
# Reverse cross-map for hypothesis:
#
# sqft_living -> price
#
# Therefore:
# price xmap sqft_living
# ------------------------------------------------------------

x <- as.numeric(
    sites$price_resid
)

y <- as.numeric(
    sites$sqft_living_resid
)

# Build the price spatial manifold
embeddings <- generateEmbedings(
    neighborMatrix,
    x,
    E
)

# Reconstruct sqft_living from the price manifold
result <- GCCMLattice(
    embeddings,
    y,
    lib_sizes,
    lib,
    pred,
    E,
    b = b,
    cores = NULL
)

write.csv(
    result,
    "E:/GeoSIMCP_Reproduce/geoconformal-main/GCCP_Seattle/outputs/gccm_author_aligned/smoke_test/old_price_xmap_sqft_living_L2086.csv",
    row.names = FALSE
)

cat(
    "\n===== R SMOKE TEST RESULT =====\n"
)

print(
    result
)

cat(
    "\nn =", n,
    "\nE =", E,
    "\nb =", b,
    "\nL =", lib_sizes,
    "\n"
)
