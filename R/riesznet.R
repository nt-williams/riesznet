#' Title
#'
#' @param data
#' @param shifted
#' @param .f
#' @param net
#' @param weights
#' @param max_lr
#' @param epochs
#' @param batch_size
#' @param valid_ratio
#' @param shuffle
#' @param patience
#' @param optimizer
#' @param verbose
#'
#' @return
#' @export
#'
#' @examples
riesznet <- function(data,
                     shifted,
                     .f,
                     net = NULL,
                     weights = NULL,
                     max_lr = 0.1,
                     epochs = 500,
                     batch_size = 32,
                     valid_ratio = 0.2,
                     shuffle = TRUE,
                     patience = 10,
                     weight_decay = 0,
                     optimizer = torch::optim_adam,
                     verbose = TRUE) {

  assert_data_frame(data)
  lapply(shifted, function(x) {
    assert_data_frame(x, nrows = nrow(data), ncols = ncol(data))
    assert_set_equal(names(x), names(data))
  })
  assert_function(.f, args = names(shifted))

  ds <- make_dataset(data, shifted, weights)

  train_ids <- sample(1:length(ds), size = (1 - valid_ratio) * length(ds))
  valid_ids <- sample(setdiff(1:length(ds), train_ids), size = valid_ratio * length(ds))

  train_ds <- torch::dataset_subset(ds, indices = train_ids)
  valid_ds <- torch::dataset_subset(ds, indices = valid_ids)

  train_dl <- torch::dataloader(train_ds, batch_size = batch_size, shuffle = shuffle)
  valid_dl <- torch::dataloader(valid_ds, batch_size = batch_size, shuffle = shuffle)

  if (is.null(net)) {
    d_in <- ncol(data)
    hidden <- ceiling(mean(c(ncol(data), 1)))
    net <- default_nn(d_in, hidden)
  }

  model <-
    luz::setup(RieszNetLearner,
               optimizer = optimizer,
               metrics = NULL) |>
    luz::set_hparams(net = net, .f = .f) |>
    luz::set_opt_hparams(weight_decay = weight_decay)

  fit <- luz::fit(
    model,
    train_dl,
    epochs = epochs,
    valid_data = valid_dl,
    verbose = verbose,
    callbacks = list(
      luz::luz_callback_early_stopping(patience = patience),
      luz::luz_callback_lr_scheduler(
        torch::lr_one_cycle,
        max_lr = max_lr,
        epochs = epochs,
        steps_per_epoch = length(train_dl),
        call_on = "on_batch_end"
      )
    )
  )

  fit$model$eval()

  out <- list(fit = fit, vars = names(data))

  class(out) <- "riesznet"
  return(out)
}
