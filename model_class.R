# 'Model' is a class which represents, trains, and tests Machine Learning regression models using K-fold cross validation.
# 
# 
# 
# 
# To create a new instance of the model class:
#   x<- Model$new(data = 
#                 ,formula = 
#                 )
# Necessary inputs for model creation:
#   data: a DataFrame object
#   formula: an object of class 'formula' which refers to the column names of 'data'
# 
# Default choices which can be over-written:
#   k.fold.number = 5 (Number of folds for k-fold validation)
#   seed.number = 35628 (seed number to induce replicability in the random 
#                        k-fold partitioning, as well as the randomness 
#                        built in to certain ML algorithms)
#   model.name = NULL (Can be used to store a character vector for the name of the algorithm)
# Class Methods:
#   create.folds(): 
#       Functionality: 
#             This function is called upon initialization to partition the data into K training
#               sets and K corresponding testing sets.
#       Returns Attributes:
#             train.folds: a list of K dataframes which are the training sets  
#             test.folds: a list of K dataframes which are the testing sets   
#             
#   train():
#       Arguments:
#             mod: the function to be called to create the model. 
#             ...: optional other parameters to be called to the model creation call
#             model.name: optional character string to store the name of the algorithm
#       Functionality: 
#             For each of the K folds, creates and trains a model using the following function call:
#               mod(<object's formula attribute>, data = <element of object's train.folds list>, ...)
#       Returns Attributes:
#             models: a list of the objects returned by the model generation call 
#             model.parameters: A named list of extra parameters put into the model generation call
#             
#   test():
#       Arguments:
#             ... : Any extra arguments needed in the call to predict(<model>, newdata = <data fold>, ...)
#       Functionality: 
#             For each of the K folds, this function makes a call to predict() to find predictions for the
#               test cases.
#             Calls calc_error().
#       Returns Attributes:
#             test.actuals : A vector of observations of the dependent variable. This is in the same order as
#                               test.predictions
#             test.predictions: A vector of predictions returned by the various trained models.
#   calc.error():
#       Functionality:
#             Takes the attributes set by test() and uses them to calculate performance metrics for the model.
#               The error metrics are calculated for the model as a whole, not 
#       Returns Attributes:
#             error.vec : A list of testing errors. Given by (actual slippage) - (predicted slippage)
#             error.metrics: A named list of test error mean, MSE, RMSE
#             error.table: A confusion matrix for all of the test cases. 
#   show():
#       Functionality:
#             Prints to standard output(usually the R console) a description of the model when the Model 
#               object itself is called 
              
            

Model <-setRefClass("Model",
                    fields = c("k.fold.number"
                               ,"seed.number"
                               ,"data"
                               ,"formula"
                               ,"n.obs"
                               ,"train.folds"
                               ,"test.folds"
                               ,"models"
                               ,"model.name"
                               ,"model.parameters"
                               ,"test.actuals"
                               ,"test.predictions"
                               ,"test.errors"
                               ,"error.vec"
                               ,"error.metrics"
                               ,"error.table")
                    , methods = list(initialize = function(k.fold.number = 5,
                                                            seed.number = 35628,
                                                            data = NULL,
                                                            formula = NULL,
                                                            model.name = NULL){
                                                  k.fold.number <<-k.fold.number
                                                  data <<-data
                                                  formula<<-formula
                                                  seed.number <<-seed.number
                                                  n.obs <<- nrow(data)
                                                  model.name <<-model.name
                                                  create.folds()
                                                }
                                      , create.folds = function(){
                                                  ss<-seq(1, n.obs)
                                                  set.seed(seed.number)
                                                  
                                                  #The call to sample() is how we inject randomness into the fold creation
                                                  ss_shuffled <- sample(ss, n.obs) 
                                                  list_of_test_folds<-list()
                                                  list_of_train_folds<-list()
                                                  for(i in seq(0,k.fold.number -1)){
                                                    
                                                    #Selects all of the indicies which when divided by K give a remainder of i
                                                    fold_filter <-ss_shuffled %% k.fold.number == i 
                                                    list_of_test_folds[[i+1]]<-data[fold_filter,]
                                                    list_of_train_folds[[i+1]]<-data[!fold_filter,]
                                                  }
                                                  train.folds <<-list_of_train_folds
                                                  test.folds <<- list_of_test_folds
                                                }
                                     , train = function( mod, ..., model.name=NULL){
                                                mod_list <- list()
                                                for(i in seq(1, k.fold.number)){
                                                  set.seed(seed.number)
                                                  new_model = mod(formula, data = train.folds[[i]], ...)
                                                  mod_list[[i]] <- new_model
                                                }
                                                models<<-mod_list
                                                if(!is.null(model.name)){
                                                  model.name <<-model.name
                                                }
                                                model.parameters <<-list(...)
                                              }
                                     , test = function(...){
                                       
                                                #as.character() applied to a formula always seems to return a list of the form
                                                #"~"  "Dep Variable Name"   "Independent Variables"
                                                dep_var_name <-as.character(formula)[2]
                                                actuals_vec <-c()
                                                pred_vec <-c()
                                                
                                                for(i in seq(1, k.fold.number)){
                                                  new_preds <-predict(models[[i]], newdata = test.folds[[i]], ...)
                                                  new_actuals <- test.folds[[i]][[dep_var_name]]
                                                  actuals_vec<-append(actuals_vec, new_actuals)
                                                  pred_vec<-append(pred_vec, new_preds)
                                                }
                                                test.actuals <<-actuals_vec
                                                test.predictions<<-pred_vec
                                                calc.error()
                                              }
                                     , calc.error = function(){
                                                error.table <<-table(test.actuals, round(test.predictions), 
                                                                     dnn = c("Actuals", "Predictions"))
                                                error.vec <<-test.actuals - test.predictions
                                                mse <- mean(error.vec**2)
                                                error.metrics <<-list("Mean" = mean(error.vec), 
                                                                      "MSE" = mse, 
                                                                      "RMSE" = sqrt(mse))
                                              }
                                     , show = function(){
                                       s <- "This is an object of class 'Model' with the following attributes: \n"
                                       if(!is.null(model.name)){
                                         s<-paste(s, "The model is built with the", model.name, 
                                                  "algorithm", "\n", sep = ' ')
                                       }
                                       s <- paste(s ,"The formua for the model is given as" , 
                                                  format(formula) , "\n", sep = ' ')
                                       s <-paste(s, "The model is validated using", as.character(k.fold.number), 
                                                 "fold validation.", sep = ' ')
                                       writeLines(s)
                                              })
                    
)

############## Use case: #######################
###  initiate, train, and test a RandomForest model given the 'iris' dataset from the datatsets package.


# library(randomForest)
# library(datasets)
# x<-datasets::iris
# frml<- Species ~ Sepal.Width + Petal.Length + Petal.Width + Sepal.Length
# b<-Model$new(formula = frml, data = x)
# b
# b$train(randomForest, mtry = 4, model.name = "Random Forest")
# b
# b$models
# b$model.parameters
# b$test()
# b$error.metrics
# b$error.table
