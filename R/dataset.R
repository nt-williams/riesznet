make_dataset <- function(x, new_x, weights = NULL) {
  self <- NULL
  ds <- torch::dataset(
    name = "riesz_dataset",
    initialize = function(x, new_x, weights) {
      if (!is.null(weights)) {
        self$weights <- as_torch(weights)
      } else {
        self$weights <- torch::torch_ones(nrow(x))
      }

      self$data <- as_torch(one_hot_encode(x))
      self$new_x <- vector("list", length(new_x))
      names(self$new_x) <- names(new_x)
      for (d in names(new_x)) {
        self$new_x[[d]] <- as_torch(one_hot_encode(new_x[[d]]))
      }
    },
    .getitem = function(i) {
      returns <- list(x = list(data = self$data[i, ],
                               new_x = list()),
                      target = list(weights = self$weights[i]))
      for (d in names(self$new_x)) {
        returns$x$new_x[[d]] <- self$new_x[[d]][i, ]
      }
      returns
    },
    .length = function() {
      self$data$size()[1]
    }
  )
  ds(x, new_x, weights)
}
