args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 2) {
  stop("usage: export_mixscale_targets.R <output.tsv> <tool1.rds> [tool2.rds ...]")
}

output <- args[[1]]
inputs <- args[-1]
targets <- character()
for (path in inputs) {
  tool <- readRDS(path)
  current <- names(tool)
  if (is.null(current)) {
    stop(sprintf("Mixscale Tool object has no target names: %s", path))
  }
  targets <- union(targets, current)
  rm(tool)
  invisible(gc())
}
targets <- sort(unique(targets), method = "radix")
temporary <- paste0(output, ".tmp")
write.table(
  data.frame(target_id = targets),
  file = temporary,
  sep = "\t",
  quote = FALSE,
  row.names = FALSE
)
if (!file.rename(temporary, output)) {
  stop(sprintf("Could not atomically publish %s", output))
}

