# Implementing Naive Bayes without importing any packages

# :) Low storage requirements, as conditional probability tables are not saved.
# :( Low efficiency, as recalculates conditional probabilities for each prediction.
# :( Not robust, as cannot handle unseen features when making new predictions.

# background ===================================================================
# Bayes Theorem: P(C|X) = P(X|C)*P(C)/P(X)
# Naive Bayes:  Calculate the posterior prob P(C|X) for each possible class
# and see which returns the highest probability.  Assign record to that class.

# P(X) will be the same for all classes, so we can exclude it from our
# calculations and still correctly find the highest probability.


# clear console
cat("\014")


# example data =================================================================

outlook <- c("sunny", "sunny", "overcast", "rain", "rain", "rain", "overcast"
             , "sunny", "sunny", "rain", "sunny", "overcast", "overcast"
             , "rain")
temp <- c("hot", "hot", "hot", "mild", "cool", "cool", "cool" , "mild", "cool"
          , "mild", "mild", "mild", "hot" , "mild")
humidity <- c("high", "high", "high", "high", "normal", "normal", "normal"
              , "high", "normal", "normal", "normal", "high", "normal", "high")
wind <- c("weak", "strong", "weak", "weak", "weak", "strong", "strong", "weak"
          , "weak", "weak", "strong", "strong", "weak", "strong")
play <- c("no", "no", "yes", "yes", "yes", "no", "yes", "no", "yes", "yes"
          , "yes", "yes", "yes", "no")

df <- data.frame(
                   Outlook=outlook
                 , Temperature=temp
                 , Humidity=humidity
                 , Wind=wind
                 , playTennis=play
                 )


# naive bayes formulae =========================================================

# P(C): prior probabilities: current info on classes
priors <- table(df$playTennis) / nrow(df)

# P(X|C): likelihoods: likelihood of feature given class
likelihood <- function(data, class, feature, value) {
  
  count_class <- data[data$playTennis==class, ]
  count_feature <- count_class[count_class[[feature]]==value, ]
  
  return(nrow(count_feature)/nrow(count_class))
}

# P(C|X): posteriors: updated probability given new info
posterior <- function(data, new_data) {
  
  results <- list()
  
  # for each class
  for (class in unique(data$playTennis)) {
    
    like <- 1
    
    # for each feature
    for (i in seq_along(new_data)) {
      # update likelihood using conditional independence assumption
      like <- like * likelihood(data, class, new_data[[i]]$col, new_data[[i]]$val)
    }
    
    # calculate bayes theorem (ignoring P(X))
    bayes <- priors[class] * like
    results[[class]] <- bayes
    
  }
  
  return(results)
}

# call naive bayes and print results
predict <- function(data, new_data, details=TRUE) {
  
  post <- posterior(data, new_data)
  
  print(paste("Prediction:", names(post[which.max(unlist(post))])))
  
  if (details == TRUE) {
      print(paste("Probabilities:", post))
  }
}


# make prediction on new data ==================================================
new_data <- list(
    list(col='Outlook', val='sunny')
  , list(col='Temperature', val='cool')
  , list(col='Humidity', val='high')
  , list(col='Wind', val='strong')
)

predict(df, new_data, TRUE)
