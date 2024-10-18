#' Title
#'
#' @param x
#' @param new_x
#' @param net
#' @param weights
#' @param .f
#' @param end_lr
#' @param batch_size
#' @param shuffle
#' @param optimizer
#'
#' @return
#' @export
#'
#' @examples
riesz_lr_finder <- function(
  x, new_x,
  net = NULL,
  weights, .f,
  end_lr = 0.1,
  batch_size = 32,
  shuffle = TRUE,
  weight_decay = 0,
  optimizer = torch::optim_adam
) {
    # check for same dimensions and names between x and new_x
    # check that the argument names in .f are the same as the names of the data frames in new_x
    ds <- make_dataset(x, new_x, weights)
    dl <- torch::dataloader(ds, batch_size = batch_size, shuffle = shuffle)
    
    if (is.null(net)) {
      d_in <- ncol(x)
      hidden <- ceiling(mean(c(ncol(x), 1)))
      net <- default_nn(d_in, hidden)
    }
    
    model <-
      luz::setup(RieszNet,
                 optimizer = optimizer,
                 metrics = NULL) |>
      luz::set_hparams(net = net, .f = .f) |>
      luz::set_opt_hparams(weight_decay = weight_decay)
    
    luz::lr_finder(model, dl, end_lr = end_lr, verbose = FALSE)
  }
