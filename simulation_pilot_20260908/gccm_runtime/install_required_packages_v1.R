options(repos=c(CRAN="https://cloud.r-project.org"), timeout=120)
userlib <- Sys.getenv("R_LIBS_USER")
stopifnot(nzchar(userlib))
dir.create(userlib, recursive=TRUE, showWarnings=FALSE)
.libPaths(c(userlib, .libPaths()))
required <- c("foreach", "doParallel")
missing <- required[!vapply(required, requireNamespace, logical(1), quietly=TRUE)]
cat("USER_LIBRARY:", userlib, "\n")
cat("REQUESTED_MISSING:", paste(missing, collapse=","), "\n")
if (length(missing)) install.packages(missing, lib=userlib, dependencies=NA, type="binary")
stopifnot(all(vapply(required, requireNamespace, logical(1), quietly=TRUE)))
for (p in c("foreach", "doParallel", "iterators", "codetools", "parallel")) {
  cat("PACKAGE:", p, " VERSION:", as.character(packageVersion(p)), " PATH:", find.package(p), "\n")
  print(packageDescription(p)[c("Version","Depends","Imports","Repository")])
}
print(sessionInfo())
