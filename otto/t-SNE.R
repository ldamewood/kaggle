# Forked from https://www.kaggle.com/forums/t/13122/visualization/68866
# which was written by https://www.kaggle.com/users/8998/piotrek
# 
# Key differences from original:
# - downsampling to 15000 train examples to run quickly on Kaggle Scripts
# - using ggplot2 for plotting

library(Rtsne)

set.seed(1)

train        <- read.csv("~/Work/kaggle/otto/data/train.csv")
test        <- read.csv("~/Work/kaggle/otto/data/test.csv")
tr_features     <- train[,c(-1, -95)]
te_features     <- test[,c(-1)]

all_features <- rbind(tr_features, te_features)

tsne <- Rtsne(as.matrix(all_features), check_duplicates = FALSE, pca = TRUE, perplexity=30, theta=0.5, dims=3)
write.csv(tsne$Y, "out.csv")

# embedding <- as.data.frame(tsne$Y)
# embedding$Class <- as.factor(sub("Class_", "", train_sample[,95]))
# 
# p <- ggplot(embedding, aes(x=V1, y=V2, color=Class)) +
#   geom_point(size=1.25) +
#   guides(colour = guide_legend(override.aes = list(size=6))) +
#   xlab("") + ylab("") +
#   ggtitle("t-SNE 2D Embedding of Products Data") +
#   theme_light(base_size=20) +
#   theme(strip.background = element_blank(),
#         strip.text.x     = element_blank(),
#         axis.text.x      = element_blank(),
#         axis.text.y      = element_blank(),
#         axis.ticks       = element_blank(),
#         axis.line        = element_blank(),
#         panel.border     = element_blank())
# 
# ggsave("tsne.png", p, width=8, height=6, units="in")
