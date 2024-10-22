#' @export
predict.riesznet <- function(fit, newdata) {
  model <- fit$fit$model
  vars <- fit$vars
  model(as_torch(one_hot_encode(newdata[, vars]))$to(device = model$device))
}

#' @export
predict.riesz_ensemble <- function(fit, newdata) {
  if (fit$discrete) {
    return(predict.riesz_torch(fit$fits[[1]], newdata))
  }

  base_preds <- sapply(fit$fits, \(x) as.numeric(predict.riesz_torch(x, newdata)))
  fit$search_grid$ensemble_weights %*% base_preds
}
