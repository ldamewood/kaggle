
library(ggplot2)
library(Rtsne)
library(lubridate)

set.seed(1)

train <- read.csv("~/Work/kaggle/revenue/data/train.csv")
tr_numeric_features <- train[,c(-1,-3,-4,-5,-43)]
test <- read.csv("~/Work/kaggle/revenue/data/test.csv")
te_numeric_features <- test[,c(-1,-3,-4,-5,-43)]
data <- rbind(tr_numeric_features, te_numeric_features)
data$OD <- as.numeric(mdy("01/01/2015")-mdy(data$Open.Date), units="days")/365
data <- data[,c(-1)]
test$revenue = 1
alldata = rbind(train, test)

tsne <- Rtsne(as.matrix(data), check_duplicates = FALSE, pca = TRUE, 
              perplexity=30, theta=0.5, dims=2)

embedding <- as.data.frame(tsne$Y)
embedding$Revenue  <- log(alldata$revenue)
embedding$Type     <- alldata$Type
embedding$CityType <- alldata[["City Group"]]

p <- ggplot(embedding[c(sample(137:100137, 1000),1:137),], aes(x=V1, y=V2, color=Revenue, shape=Type)) +
     geom_point(size=4) +
     scale_colour_gradientn(colours=c("#3288bd","#66c2a5","#abdda4","#e6f598","#fee08b","#fdae61","#f46d43","#d53e4f"), name="log Revenue") + 
     xlab("") + ylab("") +
     ggtitle("t-SNE Restaurant Visualization") + 
     theme(strip.background = element_blank(),
           strip.text.x     = element_blank(),
           axis.text.x      = element_blank(),
           axis.text.y      = element_blank(),
           axis.ticks       = element_blank(),
           axis.line        = element_blank(),
           panel.border     = element_blank())

ggsave("tsne.png", p, height=8, width=8, units="in")