#!/usr/bin/env Rscript

suppressPackageStartupMessages({
  library(Seurat)
  library(yaml)
  library(jsonlite)
})

args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 4 || args[[1]] != "--config" || args[[3]] != "--shard") {
  stop("Usage: run_shard.R --config <config.yaml> --shard <id>")
}
config <- yaml::read_yaml(args[[2]])
shard_id <- as.integer(args[[4]])
manifest <- read.delim(config$target_manifest, stringsAsFactors = FALSE)
targets <- manifest$target_label[manifest$shard_id == shard_id]
object <- readRDS(file.path(config$checkpoint_dir, sprintf("shard_%03d.rds", shard_id)))
labels <- as.character(object[[config$label_field]][, 1])
names(labels) <- colnames(object)
matrix <- LayerData(object, assay = config$expression_assay, layer = config$expression_layer)
ntc_cells <- names(labels)[labels == config$ntc_label]
neighbor_groups <- object@tools[[config$neighbor_tool]]
neighbor_cells <- do.call(rbind, lapply(neighbor_groups, function(neighbors) {
  local_ntc <- ntc_cells[ntc_cells %in% neighbors@cell.names]
  if (max(neighbors@nn.idx) > length(local_ntc)) stop("Stored neighbor index exceeds its batch-specific NTC pool")
  result <- base::matrix(local_ntc[neighbors@nn.idx], nrow = nrow(neighbors@nn.idx))
  rownames(result) <- neighbors@cell.names
  result
}))
if (anyDuplicated(rownames(neighbor_cells))) stop("Duplicated cells across stored neighbor groups")

feature_map <- NULL
if (!is.null(config$feature_map)) {
  feature_map <- read.delim(config$feature_map, stringsAsFactors = FALSE)
}
aliases <- config$target_aliases
resolve_component <- function(component) {
  symbol <- component
  if (!is.null(aliases) && component %in% names(aliases)) symbol <- aliases[[component]]
  if (is.null(feature_map)) return(intersect(symbol, rownames(matrix)))
  intersect(feature_map$feature_id[feature_map$symbol == symbol], rownames(matrix))
}

direction <- if (config$perturbation_mode == "CRISPRa") 1 else -1
records <- lapply(targets, function(target) {
  components <- strsplit(target, "+", fixed = TRUE)[[1]]
  resolved <- lapply(components, resolve_component)
  found <- resolved[lengths(resolved) > 0]
  target_cells <- names(labels)[labels == target]
  if (length(found) == 0) {
    return(data.frame(cell_id = target_cells, target_label = target, is_ntc = FALSE,
      reference_score = NA_real_, reference_status = "target_feature_missing",
      target_features_found = 0L, target_features_total = length(components),
      reference_threshold = NA_real_, stringsAsFactors = FALSE))
  }
  component_value <- function(features, cells) {
    expression <- if (length(features) == 1) as.numeric(matrix[features, ]) else colMeans(as.matrix(matrix[features, , drop = FALSE]))
    names(expression) <- colnames(matrix)
    observed <- expression[cells]
    neighbor_values <- base::matrix(expression[neighbor_cells[cells, , drop = FALSE]], nrow = length(cells))
    matched <- log1p(rowMeans(expm1(neighbor_values)))
    observed - matched
  }
  target_values <- vapply(found, component_value, numeric(length(target_cells)), cells = target_cells)
  ntc_values <- vapply(found, component_value, numeric(length(ntc_cells)), cells = ntc_cells)
  if (is.null(dim(target_values))) target_values <- matrix(target_values, ncol = 1)
  if (is.null(dim(ntc_values))) ntc_values <- matrix(ntc_values, ncol = 1)
  score <- direction * rowMeans(target_values)
  null_score <- direction * rowMeans(ntc_values)
  threshold <- unname(quantile(null_score, probs = config$control_quantile, names = FALSE, type = 7))
  data.frame(cell_id = target_cells, target_label = target, is_ntc = FALSE,
    reference_score = score, reference_status = ifelse(score > threshold, "engaged", "not_engaged"),
    target_features_found = length(found), target_features_total = length(components),
    reference_threshold = threshold, stringsAsFactors = FALSE)
})
result <- do.call(rbind, records)
output_dir <- file.path(config$output_dir, "shards", sprintf("shard_%03d", shard_id))
dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)
temporary <- file.path(output_dir, "target_expression_reference.tsv.partial")
write.table(result, temporary, sep = "\t", quote = FALSE, row.names = FALSE)
if (!file.rename(temporary, sub("\\.partial$", "", temporary))) stop("Atomic output rename failed")
write_json(list(shard = shard_id, targets = length(targets), target_cells = nrow(result),
  scorable_cells = sum(!is.na(result$reference_score)), status = "PASS"),
  file.path(output_dir, "summary.json"), pretty = TRUE, auto_unbox = TRUE)
