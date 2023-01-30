setwd("/home/bill/projects/multi-differential-calibration/overleaf/figs/")
#simulation goal, show proportional multicalibration of multicalibrated odds
set.seed(12393)
library(ggplot2)
library(tidyverse)
amc<-0.1 #alpha multicalibratoin
P_g<-seq(0.2,0.8,0.01) #
n_g<-length(P_g)
nsim=1000 #number of simulations
N=1000 #number of observations per group
R<-data.frame(matrix(data=rep(NA,n_g*nsim),nrow=n_g,ncol=nsim))#matrix for storing the the model predicted probability for each group for each sim
AME<-R #matrix for storing the absolute mean error per group for each simulation
P<-R
#below are vectors for plotting 
AME_long<-c()
R_long<-c()
P_long<-c()
for( i in 1:nsim){
  delta_sim<-rep(NA,n_g)#difference between prevalence and model prediction per group
  index<-sample(1:n_g,size=1,replace=T) #randomly sampling which group fixes the alpha-multicalibration constrain
  delta_sim[index]<-sample(c(-1,1),size=1,replace=T)*amc 
  delta_sim[which(is.na(delta_sim))]<-sample(c(-1,1),size=n_g-1,replace=T)*runif(n=n_g-1,min=0.01,max =amc) #randomly sampling the delta for the remaining groups from uniform distribution 
  r_g_sim<-P_g-delta_sim #calculating the model prediction per group
  R[,i]<-r_g_sim #storing the model prediciton per group
  R_long<-c(R_long,r_g_sim) 
  P[,i]<-P_g
  P_long<-c(P_long,P_g)
  for(j in 1:length(r_g_sim)){ #iterating through each group
    Y<-rbinom(n=N,size=1,prob=P_g[j]) #drawing the true outcomes for group j
    Y_star<-rbinom(n=N,size=1,prob=r_g_sim[j]) #drawing the predicted outcomes for group j
    AME[j,i]<-abs(mean(Y_star-Y)) #calculating absolute mean standard error for group j and storing
    AME_long<-c(AME_long,abs(mean(Y_star-Y)))
  }
}
PMC_long<-AME_long/P_long

#specific scenarios
#scenario1=abs(delta) is the same for all, 0.1, randomly positive or negative
#scenario2, abs(delta) is descending 0.1 to 0.01 while prevalence is ascending, randomly positive or negative 
#scenario3, abs(delta) is ascending (0.1 to 0.01) while prevalance is ascending
delta1=sample(c(-1,1),size=length(P_g),replace=T)*0.1
delta2<-sample(c(-1,1),size=length(P_g),replace=T)*seq(0.1,0.01,length.out=length(P_g))
delta3<-sample(c(-1,1),size=length(P_g),replace=T)*seq(0.01,0.1,length.out=length(P_g))

R_g1=P_g-delta1
R_g2=P_g-delta2
R_g3=P_g-delta3
parameters<-data.frame(P_g,R_g1,delta1,R_g2,delta2,R_g3,delta3)
alpha_calibration1<-max(abs(parameters$delta1));alpha_calibration1
alpha_calibration2<-max(abs(parameters$delta2));alpha_calibration2
alpha_calibration3<-max(abs(parameters$delta3));alpha_calibration3

truth<-data.frame(matrix(NA,N,length(P_g)))
predicted1<-truth
predicted2<-truth
predicted3<-truth
for(i in 1:ncol(truth)){
  truth[,i]<-rbinom(n=N,size=1,prob=parameters$P_g[i])
  predicted1[,i]<-rbinom(n=N,size=1,prob=parameters$R_g1[i])
  predicted2[,i]<-rbinom(n=N,size=1,prob=parameters$R_g2[i])
  predicted3[,i]<-rbinom(n=N,size=1,prob=parameters$R_g3[i])
}

results<-parameters
results$scenario1<-abs(apply(predicted1-truth,2,mean))
results$scenario2<-abs(apply(predicted2-truth,2,mean))
results$scenario3<-abs(apply(predicted3-truth,2,mean))


theme_Publication <- function(base_size=14, base_family="helvetica") {
  library(grid)
  library(ggthemes)
  (theme_foundation(base_size=base_size, base_family=base_family)
    + theme(plot.title = element_text(face = "bold",
                                      size = rel(1.2), hjust = 0.5),
            text = element_text(),
            panel.background = element_rect(colour = NA),
            plot.background = element_rect(colour = NA),
            panel.border = element_rect(colour = NA),
            axis.title = element_text(face = "bold",size = rel(1)),
            axis.title.y = element_text(angle=90,vjust =2),
            axis.title.x = element_text(vjust = -0.2),
            axis.text = element_text(), 
            axis.line = element_line(colour="black"),
            axis.ticks = element_line(),
            panel.grid.major = element_line(colour="#f0f0f0"),
            panel.grid.minor = element_blank(),
            legend.key = element_rect(colour = NA),
            legend.position = "right",
            legend.direction = "vertical",
            legend.key.size= unit(0.3, "cm"),
            legend.spacing = unit(0, "cm"),
            legend.title = element_text(),
            plot.margin=unit(c(10,5,5,5),"mm"),
            strip.background=element_rect(colour="#f0f0f0",fill="#f0f0f0"),
            strip.text = element_text(face="bold")
    ))
  
}

#creating new data frame for plotting
res<-data.frame(matrix(NA,0,4))
colnames(res)<-c("P","AME","PMC","|delta|")
temp<-data.frame(results$P_g,results$scenario1,results$scenario1/results$P_g)
temp$`|delta`<-"Fixed"
colnames(temp)<-colnames(res)
res<-rbind(res,temp)
temp<-data.frame(results$P_g,results$scenario2,results$scenario2/results$P_g)
temp$`|delta|`<-"Decreasing"
colnames(temp)<-colnames(res)
res<-rbind(res,temp)
temp<-data.frame(results$P_g,results$scenario3,results$scenario3/results$P_g)
temp$`|delta|`<-"Increasing"
colnames(temp)<-colnames(res)
res<-rbind(res,temp)
res$`|delta|`<-factor(res$`|delta|`,levels=c("Fixed","Decreasing","Increasing"),ordered=T)

p1=ggplot()+geom_point(data=res,aes(x=P,y=PMC,color=`|delta|`))+geom_line(data=res,aes(x=P,y=PMC,color=`|delta|`))+
  geom_jitter(aes(x=P_long,y=PMC_long),color="gray",alpha=0.035)+
  xlab("Prevalence")+
  ylab("Absolute Mean Error-Prevalence Ratio")+labs(title="Absolute Mean Error-Prevalence Ratio vs Prevalence \n for \u03B1-Multicalibrated Models")+theme_Publication()+
  theme(text=element_text(family="DejaVuSans"))+
  theme(legend.position = c(0.8,0.8),legend.text=element_text(size=16),legend.title=element_text(size=16),legend.title.align = 0.5,legend.background=element_rect(linetype = 1, size = 0.5, colour = 1))+
  scale_color_discrete("|\u0394|")+
  scale_x_continuous(breaks=seq(0.2,0.8,0.1))+
  scale_y_continuous(breaks=seq(0.0,0.8,0.1))

cairo_pdf("sim_plot1.pdf",width=8,height=8)
p1
dev.off()
p2=ggplot()+geom_point(data=res,aes(x=P,y=PMC,color=`|delta|`))+geom_smooth(data=res,aes(x=P,y=PMC,color=`|delta|`,fill=`|delta|`))+
  geom_jitter(aes(x=P_long,y=PMC_long),color="gray",alpha=0.035)+
  xlab("Prevalence")+
  ylab("PMC Loss")+labs(title="")+theme_Publication()+
  theme(text=element_text(family="DejaVuSans"))+
  theme(legend.position = c(0.8,0.8),legend.text=element_text(size=16),legend.title=element_text(size=16),legend.title.align = 0.5,legend.background=element_rect(linetype = 1, size = 0.5, colour = 1))+
  scale_color_discrete("|\u0394|")+
  scale_fill_discrete("|\u0394|")+
  scale_x_continuous(breaks=seq(0.2,0.8,0.1))+
  scale_y_continuous(breaks=seq(0.0,0.8,0.1))
  cairo_pdf("sim_plot2.pdf",width=8,height=8)
  p2
  dev.off()
