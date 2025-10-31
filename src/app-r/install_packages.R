pkgs <- c("readr", "dplyr", "tibble", "stringr", "tidyr")

install_if_missing <- function(p) {
  if (!requireNamespace(p, quietly = TRUE)) {
    install.packages(p, repos = "https://cloud.r-project.org")
  }
}

invisible(lapply(pkgs, install_if_missing))
message("R required packages are installed/available: ", paste(pkgs, collapse = ", "))
