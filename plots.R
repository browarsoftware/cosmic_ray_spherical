# The file visualizes the results of the evaluation

library(dplyr)
library(ggplot2)
# update path here
df = read.csv("D:\\Projects\\Python\\PycharmProjects\\tf28\\modul_c++\\siec_shower\\results_loop_all.txt")
df$th_full_abs=abs(df$th - df$th_full)
df$phi_full_abs=abs(df$phi - df$phi_full)

df$th_small_abs=abs(df$th - df$th_small)
df$phi_small_abs=abs(df$phi - df$phi_small)

df$th_sample_abs=abs(df$th - df$th_sample)
df$phi_sample_abs=abs(df$phi - df$phi_sample)

df$th_decode_abs=abs(df$th - df$th_decode)
df$phi_decode_abs=abs(df$phi - df$phi_decode)


sum(df$th_sample_abs < 10) / length(df$th_sample_abs)
sum(df$th_decode_abs < 10) / length(df$th_decode_abs)

sum(df$th_sample_abs < 5) / length(df$th_sample_abs)
sum(df$th_decode_abs < 5) / length(df$th_decode_abs)

sum(df$th_sample_abs < 1) / length(df$th_sample_abs)
sum(df$th_decode_abs < 1) / length(df$th_decode_abs)

sum(df$phi_sample_abs < 10) / length(df$th_sample_abs)
sum(df$phi_decode_abs < 10) / length(df$th_decode_abs)

sum(df$phi_sample_abs < 5) / length(df$phi_sample_abs)
sum(df$phi_decode_abs < 5) / length(df$phi_decode_abs)

sum(df$phi_sample_abs < 1) / length(df$phi_sample_abs)
sum(df$phi_decode_abs < 1) / length(df$phi_decode_abs)

agg = df %>% group_by(th, phi) %>% summarise(theta_full=mean(th_full_abs), phi_full=mean(phi_full_abs),
                                             theta_small=mean(th_small_abs), phi_small=mean(phi_small_abs),
                                             theta_sample=mean(th_sample_abs), phi_sample=mean(phi_sample_abs),
                                             theta_decode=mean(th_decode_abs), phi_decode=mean(phi_decode_abs))


#https://www.r-bloggers.com/2014/12/interactive-2d-3d-plots-with-plotly-and-ggplot2/
cols <-function(n) {
  colorRampPalette(rev(c("red4","red2","tomato2","orange","gold1","forestgreen","darkgreen","blue")))(8)
}


p = ggplot(agg, aes(x=th, y=phi, z = theta_full)) + geom_tile(aes(fill = theta_full)) + scale_fill_gradientn(colours = cols(8)) #+ stat_contour()
p + theme(legend.key.size = unit(1, 'cm')) + scale_fill_gradientn(name = 'theta (est.)', colours = cols(8), limits=c(0, 30), breaks=seq(0,150,by=5)) + xlab("theta [degrees]") + ylab("phi [degrees]") + ggtitle('800x800 image')


p = ggplot(agg, aes(x=th, y=phi, z = phi_full)) + geom_tile(aes(fill = phi_full)) + scale_fill_gradientn(colours = cols(8)) #+ stat_contour()
p + theme(legend.key.size = unit(1, 'cm')) + scale_fill_gradientn(name = 'phi (est.)', colours = cols(8), limits=c(0, 150), breaks=seq(0,150,by=10)) + xlab("theta [degrees]") + ylab("phi [degrees]") + ggtitle('800x800 image')

##############################

p = ggplot(agg, aes(x=th, y=phi, z = theta_small)) + geom_tile(aes(fill = theta_small)) + scale_fill_gradientn(colours = cols(8)) #+ stat_contour()
p + theme(legend.key.size = unit(1, 'cm')) + scale_fill_gradientn(name = 'theta (est.)',colours = cols(8), limits=c(0, 30), breaks=seq(0,150,by=5)) + xlab("theta [degrees]") + ylab("phi [degrees]") + ggtitle('80x80 image')

p = ggplot(agg, aes(x=th, y=phi, z = phi_small)) + geom_tile(aes(fill = phi_small)) + scale_fill_gradientn(colours = cols(8)) #+ stat_contour()
p + theme(legend.key.size = unit(1, 'cm')) + scale_fill_gradientn(name = 'phi (est.)', colours = cols(8), limits=c(0, 150), breaks=seq(0,150,by=10)) + xlab("theta [degrees]") + ylab("phi [degrees]") + ggtitle('80x80 image')

##############################

p = ggplot(agg, aes(x=th, y=phi, z = theta_sample)) + geom_tile(aes(fill = theta_sample)) + scale_fill_gradientn(colours = cols(8)) #+ stat_contour()
p + theme(legend.key.size = unit(1, 'cm')) + scale_fill_gradientn(name = 'theta (est.)', colours = cols(8), limits=c(0, 30), breaks=seq(0,150,by=5)) + xlab("theta [degrees]") + ylab("phi [degrees]") + ggtitle('80x80 sampled image')

p = ggplot(agg, aes(x=th, y=phi, z = phi_sample)) + geom_tile(aes(fill = phi_sample)) + scale_fill_gradientn(colours = cols(8)) #+ stat_contour()
p + theme(legend.key.size = unit(1, 'cm')) + scale_fill_gradientn(name = 'phi (est.)', colours = cols(8), limits=c(0, 150), breaks=seq(0,150,by=10)) + xlab("theta [degrees]") + ylab("phi [degrees]") + ggtitle('80x80 sampled image')

##############################

p = ggplot(agg, aes(x=th, y=phi, z = theta_decode)) + geom_tile(aes(fill = theta_decode)) + scale_fill_gradientn(colours = cols(8)) #+ stat_contour()
p + theme(legend.key.size = unit(1, 'cm')) + scale_fill_gradientn(name = 'theta (est.)', colours = cols(8), limits=c(0, 30), breaks=seq(0,150,by=5)) + xlab("theta [degrees]") + ylab("phi [degrees]") + ggtitle('80x80 sampled + E-D image')

p = ggplot(agg, aes(x=th, y=phi, z = phi_decode)) + geom_tile(aes(fill = phi_decode)) + scale_fill_gradientn(colours = cols(8))
p + theme(legend.key.size = unit(1, 'cm')) + scale_fill_gradientn(name = 'phi (est.)', colours = cols(8), limits=c(0, 150), breaks=seq(0,150,by=10)) + xlab("theta [degrees]") + ylab("phi [degrees]") + ggtitle('80x80 sampled + E-D image')

# update path here
loss = read.csv('D:\\Projects\\Python\\PycharmProjects\\tf28\\modul_c++\\siec_shower\\log_tran.csv', sep=";")
plot(loss, ylab='loss (binary crossentropy)')
points(loss, type='l', col='red')
title('Encoder-decoder training')

