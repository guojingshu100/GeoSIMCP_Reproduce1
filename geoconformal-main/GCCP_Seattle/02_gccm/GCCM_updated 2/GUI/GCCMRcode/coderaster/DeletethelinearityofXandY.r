###############################basic#####################################################################################

confidence<- function(r,n,level=0.05)
{
  z<-1/2*log((1+r)/(1-r))
  ztheta<-1/sqrt(n-3)
  
  qZ<-qnorm(1-level/2)
  
  upper<-z+qZ*ztheta
  
  lower<-z-qZ*ztheta
  
  r_upper<-(exp(2*upper)-1)/(exp(2*upper)+1)
  
  r_lower<-(exp(2*lower)-1)/(exp(2*lower)+1)
  
  
  return (cbind( r_upper, r_lower))
  
}



significance<- function(r,n)
{
  t<-r*sqrt((n-2)/(1-r*r))
  
  return (1-pt(t,n-2))*2
  
}

expandMatrix<- function(dataMatrix,lagNum)
{
  
  if(lagNum<0)
  {
    return (dataMatrix)
  }
  
  lagNum<-round(lagNum)
  
  if(lagNum>1)
  {
    dataMatrix<-expandMatrix(dataMatrix,lagNum-1)
  }
  
  ColNum<-ncol(dataMatrix)
  RowNum<-nrow(dataMatrix)
  
  dataMatrix<-rbind(rep(NA,ColNum),dataMatrix)
  dataMatrix<-rbind(dataMatrix,rep(NA,ColNum))  
  dataMatrix<-cbind(rep(NA,ColNum)+2,dataMatrix) 
  dataMatrix<-cbind(dataMatrix,rep(NA,ColNum)+2) 
  
  return(dataMatrix)
  
}


laggedVariable<-function(dataMatrix,lagNum)  
{
  ColNum<-ncol(dataMatrix)
  RowNum<-nrow(dataMatrix)
  exDataMatrix<-expandMatrix(dataMatrix,lagNum)
  
  laggedVar<-array(rep(NA,ColNum*RowNum*8*lagNum),dim=c(RowNum*ColNum,8*lagNum))
  
  for(r in 1:RowNum)
  {
    for(c in 1:ColNum)
    {
      item<-1
      exr<-r+lagNum
      exc<-c+lagNum
      #############Start from Notheast, fisr North################
      for(la in seq(lagNum,(1-lagNum)))
      {
        #####Row is  row-lagNum
        laggedVar[r,c,item]<- exDataMatrix[exr-lagNum,exc+la]
        item<-item+1
      }
      
      #############Then West################
      for(ra in seq(-lagNum,(lagNum-1)))
      {
        #####Row is  row-lagNum
        laggedVar[r,c,item]<- exDataMatrix[exr+ra,exc-lagNum]
        item<-item+1
      }
      
      #############Then South################
      
      for(la in seq(-lagNum,(lagNum-1)))
      {
        #####Row is  row-lagNum
        laggedVar[r,c,item]<- exDataMatrix[exr+lagNum,exc+la]
        item<-item+1
      }
      
      
      #############Then East################
      for(ra in seq(lagNum,(1-lagNum)))
      {
        #####Row is  row-lagNum
        laggedVar[r,c,item]<- exDataMatrix[exr+ra,exc+lagNum]
        item<-item+1
      }
      
    }
  }
  return (laggedVar)
}


laggedVariableAs2Dim<-function(dataMatrix,lagNum)  
{
  ColNum<-ncol(dataMatrix)
  RowNum<-nrow(dataMatrix)
  exDataMatrix<-expandMatrix(dataMatrix,lagNum)
  
  laggedVar<-array(rep(NA,ColNum*RowNum*8*lagNum),dim=c(RowNum*ColNum,8*lagNum))
  
  ###By row; from top to down
  for(r in 1:RowNum)
  {
    for(c in 1:ColNum)
    {
      item<-1
      exr<-r+lagNum
      exc<-c+lagNum
      #############Start from Notheast, fisr North################
      for(la in seq(lagNum,(1-lagNum)))
      {
        #####Row is  row-lagNum
        laggedVar[(r-1)*ColNum+c,item]<- exDataMatrix[exr-lagNum,exc+la]
        item<-item+1
      }
      
      #############Then West################
      for(ra in seq(-lagNum,(lagNum-1)))
      {
        #####Row is  row-lagNum
        laggedVar[(r-1)*ColNum+c,item]<- exDataMatrix[exr+ra,exc-lagNum]
        item<-item+1
      }
      
      #############Then South################
      
      for(la in seq(-lagNum,(lagNum-1)))
      {
        #####Row is  row-lagNum
        laggedVar[(r-1)*ColNum+c,item]<- exDataMatrix[exr+lagNum,exc+la]
        item<-item+1
      }
      
      
      #############Then East################
      for(ra in seq(lagNum,(1-lagNum)))
      {
        #####Row is  row-lagNum
        laggedVar[(r-1)*ColNum+c,item]<- exDataMatrix[exr+ra,exc+lagNum]
        item<-item+1
      }
      
    }
  }
  return (laggedVar)
}



distance<-function(emd1,emd2)
{
  if(is.factor(emd1))
  {
    return (distanceCat(emd1,emd2))
  }else if(is.numeric(emd1))
  {
    return (distanceCon(emd1,emd2))
  }else
  {
    return (NA)
  }
  
}



distanceCon<-function(emd1,emd2)
{
  naIn1<-which(is.na(emd1))
  naIn2<-which(is.na(emd2))
  
  emd2[naIn1]<-NA
  emd1[naIn2]<-NA
  return(sum(abs(emd1-emd2),na.rm = TRUE)) 
  
}


distanceCat<-function(emd1,emd2)
{
  
  naIn1<-which(is.na(emd1))
  naIn2<-which(is.na(emd2))
  
  emd2[naIn1]<-NA
  emd1[naIn2]<-NA
  
  table1<-table(emd1)
  table2<-table(emd2)
  
  fractions1<-as.dataframe(table1/sum(table1))
  
  fractions2<-as.dataframe(table2/sum(table2))
  
  merged<-merge(fractions1,fractions2,by.x=1,by.y=1,all=TRUE)
  merged[is.na(merged)]<-0
  
  return (sum(abs(merged[,2]-merged[,3]),na.rm = TRUE))
  
}





distanceCross<-function(trainData,embeddings,neigborNum,targetVar,xName="x",yName="y")
{
  sampleNum<-length(targetVar);
  
  values<-c()
  geoDis<-c()
  
  reOrders<-order(trainData[,targetVar])
  trainData<-trainData[reOrders,]
  
  N<-nrow(trainData) 
  
  
  for(i in  1:(N-1))
  {
    for(j in (i+1):(min(i+neigborNum, N)))
    {
      values<-c(values,abs(trainData[i,targetVar]-trainData[j,targetVar]))
      geoDis<-c(geoDis,sqrt((trainData[i,xName]-trainData[j,xName])^2+(trainData[i,yName]-trainData[j,yName])^2))
    }
    
  }
  
  resulsDistance<-data.frame(values)
  
  for(e in 1:length(embeddings) )
  {
    emd<-embeddings[[e]]
    distances<-c()
    if(is.null(dim(emd)))
    {
      emd<-emd[reOrders]
      
      N<-length(emd)
      
      for(i in  1:(N-1))
      {
        for(j in (i+1): (min(i+neigborNum, N)))
        {
          distances<-c(distances,abs(emd[i]- emd[j]))
        }
      }
      
    }else
    {
      emd<-emd[reOrders,]
      N<-nrow(emd)
      
      
      for(i in  1:(N-1))
      {
        for(j in (i+1):(min(i+neigborNum, N)))
        {
          
          distances<-c(distances,distance(emd[i,],emd[j,]))
        }
      }
    }
    
    resulsDistance<-cbind(resulsDistance,distances)
    
  }
  
  resulsDistance<-cbind(resulsDistance,geoDis)
  return (resulsDistance)
}

distanceCrossGeo<-function(sample,xName="x",yName="y")
{
  sampleNum<-nrow(sample)
  
  
  
  values<-c()
  
  for(i in  1:(sampleNum-1))
  {
    for(j in (i+1):sampleNum)
    {
      
      values<-c(values,sqrt((sample[i,xName]-sample[j,xName])^2+(sample[i,yName]-sample[j,yName])^2))
    }
  }
  
  return (values)
}

distanceBtwGeo<-function(test,train, xName="x",yName="y")
{
  
  values<-c()
  
  for(i in 1:nrow(train))
  {
    for(j in 1:nrow(test))
    {
      values<-c(values,sqrt((train[i,xName]-test[j,xName])^2+(train[i,yName]-test[j,yName])^2))
    }
  }
  
  return (values)
}


distanceBtw<-function(predEmd,embeddings)
{
  resulsDistance<-c()
  for(e in 1:length(embeddings) )
  {
    emd<-embeddings[[e]]
    distances<-c()
    if(is.null(dim(emd)))
    {
      for(i in  1:(length(emd)))
      {
        distances<-c(distances,abs(emd[i]- predEmd[[e]]))
      }
      
    }else
    {
      for(i in  1:(nrow(emd)))
      {
        distances<-c(distances, distance(emd[i,],predEmd[[e]]))
      }
      
    }
    resulsDistance<-cbind(resulsDistance,distances)
    
  }
  return (resulsDistance)
}


getGridEmbedings<-function(data,nrow,ncol,byrow,atts,lags)
{
  embeddings<-list()
  
  num<-0
  for(a in 1:length(atts))
  {
    for(l in lags)  
    {
      
      num<-num+1
      if(l==0)
      {
        embeddings[[num]]<-data[,atts[a]]
      }else
      {
        embeddings[[num]]<-laggedVariableAs2Dim(matrix(data[,atts[a]],nrow=nrow,ncol=ncol,byrow=byrow),l)
        
      }
      
    }
  }
  return (embeddings)
}


getSampleEmbedings<-function(ids,embeddings)
{
  sampleEmbedings<-list()
  for(e in 1:length(embeddings))
  {
    tmpEmbedings<-embeddings[[e]]
    
    if(is.null(dim(tmpEmbedings)))
    {
      sampleEmbedings[[e]]<-tmpEmbedings[ids]
    }else
    {
      sampleEmbedings[[e]]<-tmpEmbedings[ids,]
    }
  }
  
  return (sampleEmbedings)
}

normalize <- function(x) 
{
  return((x - min(x)) / (max(x) - min(x)))
}



dominate<-function(solutions)
{
  paretos<-solutions[1,]
  paretoIndex<-c(1)
  
  for(i in 2:nrow(solutions))
  {
    removes<-c()
    added<-TRUE
    s<-solutions[i,]
    for(j in 1:nrow(paretos))
    {
      p<-paretos[j,]
      if(all((p-s)>0))                          #all()   any()
      {
        removes<-c(removes,j)
        
      }else if(any((p-s)>0))
      {
        next
      }else
      {
        added<-FALSE
        break
      }
    }
    if(length(removes)>0)
    {
      paretos<-paretos[-removes,]
      paretoIndex<-paretoIndex[-removes]
    }
    if(added)
    {
      paretos<-rbind(paretos, s)
      paretoIndex<-c(paretoIndex,i)
    }
  }
  return (paretoIndex)
}

distanceCrossSign<-function(data,targetVars)
{
  results<-c()
  n<-nrow(data)
  
  for(targetV in targetVars)
  {
    if(is.numeric(data[,targetV]))
    {
      values<-c()
      for(i in  1:n)
      {
        for(j in 1:n)
        {
          values<-c(values, (data[i,targetV]- data[j,targetV]))
        }
      }
      results<-cbind(results,values)
    }else
    {
      values1<-c()
      values2<-c()
      for(i in  1:n)
      {
        for(j in 1:n)
        {
          values1<-c(values1,data[i,targetV])
          values2<-c(values2,data[j,targetV])
        }
      }
      results<-cbind(results,values1,values2)
    }
    
  }
  
  results<-as.data.frame(results)
  
  names(results)<-targetVars
  
  return (results)
  
  #  distance<-function(emd1,emd2)
  #  {
  #    if(is.factor(emd1))
  #    {
  #      return (distanceCat(emd1,emd2))
  #    }else if(is.numeric(emd1))
  #    {
  #      return (distanceCon(emd1,emd2))
  #    }else
  #    {
  #      return (NA)
  #    }
  
  #  }
  
}


distanceSign<-function(former,latter,targetVars)
{
  results<-c()
  
  for(f in 1:nrow(former) )
  {
    for(l in 1:nrow(latter) )
    {
      values<-c()
      for(targetV in targetVars)
      {
        if(is.numeric(former[f,targetV]))
        {
          values<-c(values,former[f,targetV]-latter[l,targetV])
        }else
        {
          values<-c(values,former[f,targetV],latter[l,targetV])
        }
      }
      
      results<-rbind(results,values)
    }
  }
  
  results<-as.data.frame(results)
  
  names(results)<-targetVars
  
  return (results)
  
}


compute_stats <- function(obs, pred)
{
  # computes performance metrics for how well predictions match observations
  # obs = vector of observations
  # pred = vector of prediction
  
  N = sum(is.finite(obs) & is.finite(pred))
  rho = cor(obs, pred, use = "pairwise.complete.obs")
  mae = mean(abs(obs-pred), na.rm = TRUE)
  rmse = sqrt(mean((obs-pred)^2, na.rm = TRUE))
  return(data.frame(N = N, rho = rho, mae = mae, rmse = rmse))
}



#############################GCCM###################################################################################
# GCCMSingle(xEmbedings,yPred,lib_size,pred,totalRow,totalCol,b,cores)
GCCMSingle<-function(xEmbedings,yPred,lib_size,pred,totalRow,totalCol,b,winStepRatio,cores=NULL,dir,validRatio)
{
  x_xmap_y <- data.frame()
  
  
  if(is.null(cores))
  {
    
    for(r in seq(1,(totalRow-lib_size+1),round(1+winStepRatio* lib_size)))
    {
      
      x_xmap_y<-rbind(x_xmap_y,GCCMSingleInner(xEmbedings,yPred,lib_size,pred,totalRow,totalCol,b,r,winStepRatio,dir,validRatio))
    }
    
  }else
  {
    cl <- makeCluster(cores)
    registerDoParallel(cl)
    clusterExport(cl,deparse(substitute(GCCMSingleInner)))
    clusterExport(cl,deparse(substitute(locate)))
    clusterExport(cl,deparse(substitute(projection)))
    clusterExport(cl,deparse(substitute(distance_Com)))
    clusterExport(cl,deparse(substitute(compute_stats)))
    
    x_xmap_y <- foreach(r=seq(1,(totalRow-lib_size+1),round(1+winStepRatio* lib_size)), .combine='rbind') %dopar% GCCMSingleInner(xEmbedings,yPred,lib_size,pred,totalRow,totalCol,b,r,winStepRatio,dir,validRatio)
    stopCluster(cl)
  }

  
  return(x_xmap_y)
}

# GCCMSingleInner(xEmbedings,yPred,lib_size,pred,totalRow,totalCol,b,r)
GCCMSingleInner<-function(xEmbedings,yPred,lib_size,pred,totalRow,totalCol,b,r,winStepRatio,dir,validRatio)
{
  x_xmap_y <- data.frame()
  
  for(c in seq(1,(totalCol-lib_size+1),round(1+winStepRatio* lib_size)))
  {
    
    
    pred_indices <- rep.int(FALSE, times = totalRow*totalCol)
    lib_indices<- rep.int(FALSE, times = totalRow*totalCol)
    
    pred_indices[locate(pred[,1],pred[,2],totalRow,totalCol) ]<-TRUE
    
    pred_indices[which(is.na(yPred)) ]<-FALSE
    
    
    lib_rows<-seq(r,(r+lib_size-1))
    lib_cols<-seq(c,(c+lib_size-1))
    
    lib_ids<-merge(lib_rows,lib_cols)
    
    
    lib_indices[locate(lib_ids[,1],lib_ids[,2],totalRow,totalCol)]<-TRUE
    
    
    if(length(which(is.na(yPred[which(lib_indices)]))) > ((lib_size*lib_size)/2))
    {
      next
    }
    
    # run cross map and store results
    results <-  projection(xEmbedings,yPred,lib_indices ,pred_indices,b,dir,validRatio)
    
    
    x_xmap_y <- rbind(x_xmap_y, data.frame(L = lib_size, rho = results$stats$rho)) 
  }
  
  return(x_xmap_y)
}


# GCCM(xMatrixM, yMatrixM, lib_sizes, lib, pred, E,cores=32)
GCCM<-function(xMatrix, yMatrix, lib_sizes, lib, pred, E, tau = 1, b = E+2, winStepRatio=0,cores=NULL,dir=0,validRatio=0)
{
  
  ### xMatrix: Two dimension matrix of X variable
  ### yMatrix: Two dimension matrix of Y variable
  ### lib_sizes: the sizes of libs to be calculated, they will appears at the horizontal axis
  ### lib: is not used in this version. It is kept for users to define a certain part of the matrix to be used in the cross mapping prediction 
  #         In this version, we use all spatial units of xMatrix and yMatrix by default.
  ### pred: indices of spatial units to be predicted. Due to spatial correlation, the prediction precision of nearby units are  
  #         almost the same, so we can skip some spatial units to  speed up the algorithm
  ### E: is the number of spatial lags used to construct the embeding
  ### b: number of nearest neighbors to use for prediction. We set it default to E+2 according to 
  #               suggestions from Katharina M Bracher，Stuart King and Ian Maccormick of University of Edinburgh, Edinburgh.
  #               Here we acknowledge their contributions 
  ### tau:is the step of spatial lags. If tau=1, the first ring around the focus unit is used as the when E=1; while if tau=2,the 
  #       second ring around the focus unit is used as the when E=1  
  ### winStepRatio: is a speedup parameter. In each prediction, a sliding window with the lib_size is used to confine the number of points in the state space.
  #                 If the matrix is very large, it is time consuming. We can increase the sliding step with winStepRatio. The winStepRatio will be multiplied with 
  #                 the width/height to set the sliding steps  
  #                 
  ### cores: the number of cores can be used for parallel computing
  ### dir: direction parameter for anisotropy. It used to select spatial units from  spatial lags to be used to reconstruct the state sapce. That means only spatial lags 
  ###      in the direction defined by dir take into the reconstruction of the state space. 
  ###      dir=0, all directions are considered. dir=1, Northeast; dir=2, North; dir=3, Northwest; dir=4, West; dir=5, Southwest; dir=6, South; dir=7, Southeast; dir=8, East;  
  ###      dir can also be a vector with more than one directions. For example, dir=c(1,2), then both Northeast and North will be used; dir=c(1,5),Northeast and Southwest
  ### validRatio: is the parameters to handle NA values. When the study area have too many NA (or Nodata) values, we would get NA results. To handle the NA values，we will 
  #               neglect the NA values of target variables. But it will also causes NA results or unstable predictions. So the validRatio is used to expand the neighbors 
  #               to farther sate points: maxDistacne+meanDistance*validRatio，where maxDistacne is the largest distance of the closest b neighbors, the meanDistance is the 
  #               average distance f the closest b neighbors. If validRatio=0.01, then the candidates set were expanded to 0.01*meanDistance. Then after removing NA values of the 
  #               closest b neighbors, farther sate points in candidates set could be used to replenish
  
  
  imageSize<-dim(xMatrix)
  totalRow<-imageSize[1]
  totalCol<-imageSize[2]
  
  yPred<- as.vector(t(yMatrix))
  
  xEmbedings<-list()
  xEmbedings[[1]]<- as.vector(t(xMatrix))
  
  for(i in 1:E)
  {
    xEmbedings[[i+1]]<-laggedVariableAs2Dim(xMatrix, i*tau)  #### row first
  
  }
  
  x_xmap_y <- data.frame()

  for(lib_size in lib_sizes)
  {
   
     x_xmap_y<-rbind(x_xmap_y,GCCMSingle(xEmbedings,yPred,lib_size,pred,totalRow,totalCol,b,winStepRatio,cores,dir,validRatio))
      
  }
  
  return (x_xmap_y)
}

locate<-function(curRow,curCOl,totalRow,totalCol)
{
  return ((curRow-1)*totalCol+curCOl) 
}

# projection(xEmbedings,yPred,lib_indices ,pred_indices,b)
projection<-function(embedings,target,lib_indices, pred_indices,num_neighbors,dir,validRatio=0)
{
  makepred <- rep.int(NaN, times = length(target))
  
  for(p in which (pred_indices))
  {
    temp_lib <- lib_indices[p]
    lib_indices[p] <- FALSE
    
    libs <- which(lib_indices)
    
    distances<-distance_Com(embedings,libs,p,dir)
    #distances<-colMeans(distances)
    distances_sorted<-sort(distances,na.last = TRUE)
    nearest_distances<-distances_sorted[1:num_neighbors]
    meandis<-mean(nearest_distances,na.rm=TRUE)
    maxdis<-max(nearest_distances,na.rm = TRUE)
    distances_Ex<-maxdis+validRatio*meandis
    distances[distances>distances_Ex]<-NA
    
    
    
    ###########deal with Na###########
    target4Lib<-target[libs]
    distances[which(is.na(target4Lib))]<-NA
    ####################
    # find nearest neighbors
    # neighbors <- order(distances)[1:num_neighbors]
    neighbors <- order(distances,na.last = TRUE)[1:num_neighbors]
    min_distance <- distances[neighbors[1]]
    if(is.na(min_distance))
    {
      
      lib_indices[p] <- temp_lib 
      
      next
    }
    # compute weights
    if(min_distance == 0) # perfect match
    {
      weights <- rep.int(0.000001, times = num_neighbors)
      weights[distances[neighbors] == 0] <- 1
      flagstore<-c()
      for (flag in 1:num_neighbors)
      {
        if(is.na(target[libs[neighbors]][flag]))# if(is.na(test[flag]))
        {
          next
        }
        flagstore<-c(flagstore,flag)
        
      }
      numerator=0
      denominator=0
      if(!(is.null(flagstore)))
      {
        for (calcuflag in flagstore)
        {
          numerator<-numerator+(weights[calcuflag]*target[libs[neighbors]][calcuflag])
          
          denominator<-denominator+weights[calcuflag]
          
        }
      }
    }else{
     
      if(length(which(is.na(distances[neighbors])))!=0)
      {
        
        weights <- exp(-distances[neighbors]/min_distance)
        weights[weights < 0.000001] <- 0.000001
        
        Truenieghbors<-neighbors[which(!is.na(weights))]
      
       
        flagstore<-c()
        for (flag in 1:length(Truenieghbors))
        {
          
          if(is.na(target[libs[Truenieghbors]][flag]))# if(is.na(test[flag]))
          {
            next
          }
          flagstore<-c(flagstore,flag)
          
        }
        
        numerator=0
        denominator=0
       
        if(!(is.null(flagstore)))
        {
          for (calcuflag in flagstore) 
          {
            numerator<-numerator+(weights[calcuflag]*target[libs[neighbors]][calcuflag])
            
            denominator<-denominator+weights[calcuflag]
            
           
            
          }
        }
      }else{
       
        weights <- exp(-distances[neighbors]/min_distance)
        weights[weights < 0.000001] <- 0.000001
        flagstore<-c()
       
        for (flag in 1:num_neighbors)
        {
          if(is.na(target[libs[neighbors]][flag]))# if(is.na(test[flag]))
          {
            next
          }
          flagstore<-c(flagstore,flag)
          
        }
        numerator=0
        denominator=0
        if(!(is.null(flagstore)))
        {
          for (calcuflag in flagstore) 
          {
            numerator<-numerator+(weights[calcuflag]*target[libs[neighbors]][calcuflag])
           
            denominator<-denominator+weights[calcuflag]
           
          } 
        }
      }
      
    }
    
    
    
    makepred[p] <-numerator/denominator
    
    lib_indices[p] <- temp_lib 
    numerator=NaN
    denominator=NaN
  }
  
  # return output & stats
  return(list(pred = makepred, stats = compute_stats(target[pred_indices], makepred[pred_indices])))
  
  
}


# distance_Com(embedings,libs,p)
distance_Com<-function(embeddings,libs,p,dir=0)
{
  distances<-c()
  
  emd<-embeddings[[1]]
  
  distances<-cbind(distances,abs(emd[libs]-emd[p]))
  
  
  
  for(e in 2:length(embeddings))
  {
    emd<-embeddings[[e]]
    
    dirIndices<-seq(1:dim(emd)[2])
    
    if(dir[1]!=0)
    {
      allDirIndices<-seq(1,by=(e-1),length.out=8)
      dirIndices<-allDirIndices[dir]
    }
    
    
    #q <- matrix(rep(emd[p], length(libs)), nrow = length(libs), byrow = T)
    p_matrix<-matrix(emd[p,dirIndices], nrow = length(libs), ncol = length(dirIndices), byrow = TRUE )
    distances<-cbind(distances,abs(emd[libs,dirIndices]-p_matrix))
    
    # distances<-cbind(distances,abs(emd[libs,]-emd[p,]))
    
  }
  
  return (rowMeans(distances,na.rm=TRUE))
  
}

###################################GCCM raster############################################################
library(parallel)
library(foreach)
library(doParallel)
library(rgdal)

# GCCM(xMatrix, yMatrix, lib_sizes, lib, pred, E,tau = 1,b=E+2,winStepRatio = 0,cores=16,dir=0)
GCCMrastermainfunction<-function(minlibtif,maxlibtif,intervalibtif,E,cores,minprerow,intervalprerow,minprecol,intervalprecol,tau,b,winStepRatio,dir,validRatio,selectedImagePath1,selectedImagePath2,savePath)
{
  
  yImage<-readGDAL(selectedImagePath1)       #read the effect variable
  yPath<-paste("path is",selectedImagePath1)
  yMatrix<-as.matrix(yImage)
  
  lib_sizes<-seq(minlibtif,maxlibtif,intervalibtif)  # library sizes, will be the horizontal ordinate  of the reulst plot.Note here the lib_size is the window size
  # The largest value ('to' parameter) can be set to the largest size of immage (the minor of width and length)
  # the 'by' can be set by takning accout to the computation time
  
  lib<-NULL
  
  imageSize<-dim(yMatrix)   
  totalRow<-imageSize[1]     #Get the row number of image
  totalCol<-imageSize[2]     #Get the collumn number of image
  
  predRows<-seq(minprerow,totalRow,intervalprerow)
  
  predCols<-seq(minprecol,totalCol,intervalprecol)   #To save the computation time, not every pixels are predict. The results are almost the same due to the spatial autocorrealtion 
  #If computation resources are enough, this filter can be ignored 
  
  pred<-merge(predRows,predCols)
  
  #plot(yImage)
  coodsX<-seq(1,totalRow)
  coodsY<-seq(1,totalCol)
  
  coords<-merge(coodsX,coodsY)
  
  colnames(coords)<-c("coordX","coodY")   

  y<-as.vector(yMatrix)

  lmModel<-lm(y ~ coordX+coodY, cbind(y,coords))  #remove the linear trend of y
  prediction<-predict(lmModel,coords)
  y<-y-prediction
  yMatrixM<-matrix(y,nrow = totalRow, ncol = totalCol)
  
  xImage<-readGDAL(selectedImagePath2)     #read the casue variable
  xPath<-paste("path is",selectedImagePath2)
  
  xMatrix<-as.matrix(xImage)
  
  
  x<-as.vector(xMatrix)

  lmModel<-lm(x ~ coordX+coodY, cbind(x,coords))    #remove the linear trend of x
  prediction<-predict(lmModel,coords)
  x<-x-prediction
  xMatrixM<-matrix(x,nrow = totalRow, ncol = totalCol)

  
  startTime<-Sys.time()
  
  # x_xmap_y <- GCCM(xMatrixM, yMatrixM, lib_sizes, lib, pred, E,cores) # predict y with x
  # y_xmap_x <- GCCM(yMatrixM, xMatrixM, lib_sizes, lib, pred, E,cores) #predict x with y
  # GCCM<-function(xMatrix, yMatrix, lib_sizes, lib, pred, E, tau = 1, b = E+2, winStepRatio=0,cores=NULL,dir=0,validRatio=0)
  x_xmap_y <- GCCM(xMatrixM, yMatrixM, lib_sizes, lib, pred, E,tau ,b,winStepRatio,cores,dir,validRatio)
  y_xmap_x <- GCCM(yMatrixM, xMatrixM, lib_sizes, lib, pred, E,tau ,b,winStepRatio,cores,dir,validRatio)
  
  endTime<-Sys.time()
  
  print(difftime(endTime,startTime, units ="mins"))
  
  x_xmap_y$L <- as.factor(x_xmap_y$L)
  x_xmap_y_means <- do.call(rbind, lapply(split(x_xmap_y, x_xmap_y$L), function(x){max(0, mean(x$rho,na.rm=TRUE))}))
  #calculate the mean of prediction accuray, measure by Pearson correlation
  
  y_xmap_x$L <- as.factor(y_xmap_x$L)
  y_xmap_x_means <- do.call(rbind, lapply(split(y_xmap_x, y_xmap_x$L), function(x){max(0, mean(x$rho,na.rm=TRUE))}))
  #calculate the mean of prediction accuray, measure by Pearson correlation
  
  
  predIndices<-locate(pred[,1],pred[,2],totalRow,totalCol)
  yPred<- as.array(t(yMatrix))
  predicted<-na.omit(yPred[predIndices])
  
  x_xmap_y_Sig<- significance(x_xmap_y_means,length(predicted))    #Test the significance of the prediciton accuray
  y_xmap_x_Sig<- significance(y_xmap_x_means,length(predicted))     #Test the significance of the prediciton accuray
  
  
  x_xmap_y_interval<- confidence(x_xmap_y_means,length(predicted)) #calculate the  95%. confidence interval  of the prediciton accuray
  colnames(x_xmap_y_interval)<-c("x_xmap_y_upper","x_xmap_y_lower")
  
  y_xmap_x_interval<- confidence(y_xmap_x_means,length(predicted)) #calculate the  95%. confidence interval  of the prediciton accuray
  colnames(y_xmap_x_interval)<-c("y_xmap_x_upper","y_xmap_x_lower")
  
  # save(yPath,xPath,minlibtif,maxlibtif,intervalibtif,E,cores,minprerow,intervalprerow,minprecol,intervalprecol,tau,b,winStepRatio,dir,validRatio,
  #      lib_sizes,x_xmap_y_means,y_xmap_x_means,x_xmap_y_Sig,y_xmap_x_Sig,x_xmap_y_interval,y_xmap_x_interval,file = savePath)
  variables <- list(
    xPath = xPath,
    yPath = yPath,
    minlibtif = minlibtif,
    maxlibtif = maxlibtif,
    intervalibtif = intervalibtif,
    E = E,
    cores = cores,
    minprerow = minprerow,
    intervalprerow = intervalprerow,
    minprecol = minprecol,
    intervalprecol = intervalprecol,
    tau = tau,
    b = b,
    winStepRatio = winStepRatio,
    dir = dir,
    validRatio = validRatio,
    lib_sizes = paste(lib_sizes, collapse = ", "),  # 如果是向量，转化为字符串
    x_xmap_y_means = paste(x_xmap_y_means, collapse = ", "),
    y_xmap_x_means = paste(y_xmap_x_means, collapse = ", "),
    x_xmap_y_Sig = paste(x_xmap_y_Sig, collapse = ", "),
    y_xmap_x_Sig = paste(y_xmap_x_Sig, collapse = ", "),
    x_xmap_y_interval = paste(x_xmap_y_interval, collapse = ", "),
    y_xmap_x_interval =paste( y_xmap_x_interval, collapse = ", ")
  )
  
  
  output_file <- savePath
  sink(file(output_file, encoding = "UTF-8"))
  

  for (var_name in names(variables)) {
    cat(var_name, ":", variables[[var_name]], "\n")
  }
  
  sink()  
  
  return() 
}
