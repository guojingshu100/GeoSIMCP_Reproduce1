##############################GCCM basic######################################################
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


############################GCCM4Lattice####################################################

GCCMSingle4Lattice<-function(x_vectors,y,lib_indices,lib_size,max_lib_size,possible_lib_indices,pred_indices,b)
{
  n <- NROW(x_vectors) 
  x_xmap_y <- data.frame()
  
  if(lib_size == max_lib_size) # no possible lib variation if using all vectors
  {
    lib_indices <- rep.int(FALSE, times = n)
    lib_indices[possible_lib_indices] <- TRUE
    
    # run cross map and store results
    results <- simplex_projection(x_vectors, y, lib_indices, pred_indices, b)
    x_xmap_y <- rbind(x_xmap_y, data.frame(L = lib_size, rho = results$stats$rho))
  }
  else
  {
    for(start_lib in 1:max_lib_size)
    {
      lib_indices <- rep.int(FALSE, times = n)
      # setup changing library
      if(start_lib+lib_size-1 > max_lib_size) # loop around to beginning of lib indices
      {
        lib_indices[possible_lib_indices[start_lib:max_lib_size]] <- TRUE
        num_vectors_remaining <- lib_size - (max_lib_size-start_lib+1)
        lib_indices[possible_lib_indices[1:num_vectors_remaining]] <- TRUE
      }
      else
      {
        lib_indices[possible_lib_indices[start_lib:(start_lib+lib_size-1)]] <- TRUE
      }
      
      # run cross map and store results
      results <- simplex_projection(x_vectors, y, lib_indices, pred_indices, b)
      x_xmap_y <- rbind(x_xmap_y, data.frame(L = lib_size, rho = results$stats$rho)) 
    }
  }
  return(x_xmap_y)
}




laggedVar4Lattic<- function (spNeighbor,lagNum)
{
  
  lagSpNeighbor<- spNeighbor
  
  if (lagNum<1)
  {
    
    return (NULL)
  }else if(lagNum>1)
  {
    #preSpNeighbor<-spNeighbor
    
    curSpNeighbor<-spNeighbor
    
    for(lag in 1:(lagNum-1) )
    {
      
      preSpNeighbor<-curSpNeighbor
      
      for(i in 1: length(preSpNeighbor))
      {
        curChain <-preSpNeighbor[[i]]
        
        newRings<-curChain
        
        for(neigh in 1: length (curChain))
        {
          nextChainID<-curChain[neigh]
          
          
          if(nextChainID>0)
          {
            nextChain<- spNeighbor[[nextChainID]]
            
            newRings<-unique(c(newRings, nextChain))
          }
          
          
        }
        
        curSpNeighbor[[i]]<-newRings
      }
      
    }
    
    for(i in 1: length(preSpNeighbor))
    {
      
      lagSpNeighbor[[i]]<- curSpNeighbor[[i]][!(curSpNeighbor[[i]] %in% c(preSpNeighbor[[i]],i))]
      
    }
    
  }
  
  return(lagSpNeighbor)
  
}

generateEmbedings<- function (neighborMatrix,x,E)
{
  n <- NROW(x)
  xEmbedings <- matrix(NaN, nrow = n, ncol = E)
  
  
  for(lagNum in 1:E)
  {
    laggedResults<-laggedVar4Lattic(neighborMatrix,lagNum)
    
    for(l in 1:length (laggedResults))
    {
      neighbors<-laggedResults[[l]]
      
      neighborValues<-x[neighbors]
      
      xEmbedings[l,lagNum]<-mean(neighborValues)
      
    }
    
  }
  
  return (xEmbedings)
}


GCCMLattice <- function(x_vectors, y, lib_sizes, lib, pred, E, tau = 1, b = E+2,cores=NULL)
{
  # do convergent cross mapping using simplex projection
  # x           = time series to cross map from
  # y           = time series to cross map to
  # lib_sizes   = vector of library sizes to use
  # lib         = matrix (n x 2) using n sequences of data to construct libraries
  # pred        = matrix (n x 2) using n sequences of data to predict from
  # E           = number of spatial lags for the attractor reconstruction
  # tau         = time lag for the lagged-vector construction
  # b           = number of nearest neighbors to use for prediction. We set it default to E+2 according to 
  #               suggestions from Katharina M Bracher，Stuart King and Ian Maccormick of University of Edinburgh, Edinburgh.
  #               Here we acknowledge their contributions 
  
  n <- NROW(x_vectors)
  pred <- matrix(pred, ncol = 2, byrow = TRUE)
  lib <- matrix(lib, ncol = 2, byrow = TRUE)
  
  
  
  # setup pred_indices
  pred_indices <- rep.int(FALSE, times = n)
  for(i in 1:NROW(pred))
  {
    row_start <- pred[i, 1] + (E-1)*tau
    row_end <- pred[i, 2]
    if(row_end > row_start)
      pred_indices[row_start:row_end] <- TRUE
  }
  
  # setup lib_indices
  lib_indices <- rep.int(FALSE, times = n)
  for(i in 1:NROW(lib))
  {
    row_start <- lib[i, 1] + (E-1)*tau
    row_end <- lib[i, 2]
    if(row_end > row_start)
      lib_indices[row_start:row_end] <- TRUE
  }
  max_lib_size <- sum(lib_indices) # maximum lib size
  possible_lib_indices <- which(lib_indices) # create vector of usable lib vectors
  
  # make sure max lib size not exceeded and remove duplicates
  lib_sizes <- unique(pmin(max_lib_size, lib_sizes))
  
  x_xmap_y <- data.frame()
  
  
  if(is.null(cores))
  {
    for(lib_size in lib_sizes)
    {
      
      x_xmap_y<-rbind(x_xmap_y,GCCMSingle4Lattice(x_vectors,y,lib_indices,lib_size,max_lib_size,possible_lib_indices,pred_indices,b))
      
    }
  }else
  {
    cl <- makeCluster(cores)
    registerDoParallel(cl)
    clusterExport(cl,deparse(substitute(GCCMSingle4Lattice)))
    #clusterExport(cl,deparse(substitute(locate)))
    clusterExport(cl,deparse(substitute(simplex_projection)))
    #clusterExport(cl,deparse(substitute(distance_Com)))
    clusterExport(cl,deparse(substitute(compute_stats)))
    
    x_xmap_y <- foreach(lib_size=lib_sizes, .combine='rbind') %dopar% GCCMSingle4Lattice(x_vectors,y,lib_indices,lib_size,max_lib_size,possible_lib_indices,pred_indices,b)
    
    
    
    stopCluster(cl)
  }
  
  
  return(x_xmap_y)
}

univariate_SSR <- function(data, lib, pred, E, tau = 1, tp = 1, b = E+1)
{
  # do univariate prediction using simplex projection
  # data = time series
  # lib = matrix (n x 2) using n sequences of data for library
  # pred = matrix (n x 2) using n sequences of data to predict from
  # E = number of dimensions for the attractor reconstruction
  # tau = time lag for the lagged-vector construction
  # tp = time step for future predictions
  # b = number of nearest neighbors to use for prediction
  
  n <- NROW(data)
  lib <- matrix(lib, ncol = 2)
  pred <- matrix(pred, ncol = 2)
  
  # setup vectors
  vectors <- matrix(NaN, nrow = n, ncol = E)
  lag <- 0
  for (i in 1:E)
  {
    vectors[(lag+1):n,i] <- data[1:(n-lag)]
    lag <- lag + tau
  }
  
  # setup lib_indices
  lib_indices <- rep.int(FALSE, times = n)
  for(i in 1:NROW(lib))
  {
    row_start <- lib[i, 1] + (E-1)*tau
    row_end <- lib[i, 2] - tp
    if(row_end > row_start)
      lib_indices[row_start:row_end] <- TRUE
  }
  
  # setup pred_indices
  pred_indices <- rep.int(FALSE, times = n)
  for(i in 1:NROW(pred))
  {
    row_start <- pred[i, 1] + (E-1)*tau
    row_end <- pred[i, 2] - tp
    if(row_end > row_start)
      pred_indices[row_start:row_end] <- TRUE
  }
  
  # setup target
  target <- rep.int(NaN, times = n)
  target[1:(n-tp)] <- data[(1+tp):n]
  
  return(simplex_projection(vectors, target, lib_indices, pred_indices, b))
}

simplex_projection <- function(vectors, target, lib_indices, pred_indices, num_neighbors)
{
  # do simplex projection
  # vectors = reconstructed state-space (each row is a separate vector/state)
  # target = time series to be used as the target (should line up with vectors)
  # lib_indices = vector of T/F values (which states to include when searching for neighbors)
  # pred_indices = vector of T/F values (which states to predict from)
  # num_neighbors = number of neighbors to use for simplex projection
  
  # setup output
  pred <- rep.int(NaN, times = length(target))
  
  # make predictions
  for(p in which(pred_indices))
  {
    temp_lib <- lib_indices[p]
    lib_indices[p] <- FALSE
    libs <- which(lib_indices)
    
    # compute distances
    q <- matrix(rep(vectors[p,], length(libs)), nrow = length(libs), byrow = T)
    
    dis<-(vectors[libs,] - q)^2 
    distances <- sqrt(rowSums(dis,na.rm = TRUE)/rowSums(!is.na(dis)))  ###Bingbo added, to handle NA 
    
    # find nearest neighbors
    neighbors <- order(distances)[1:num_neighbors]
    min_distance <- distances[neighbors[1]]
    
    # compute weights
    if(min_distance == 0) # perfect match
    {
      weights <- rep.int(0.000001, times = num_neighbors)
      weights[distances[neighbors] == 0] <- 1
    }
    else
    {
      weights <- exp(-distances[neighbors]/min_distance)
      weights[weights < 0.000001] <- 0.000001
    }
    total_weight <- sum(weights)
    
    # make prediction
    pred[p] <- (weights %*% target[libs[neighbors]]) / total_weight
    
    lib_indices[p] <- temp_lib 
  }
  
  # return output & stats
  return(list(pred = pred, stats = compute_stats(target[pred_indices], pred[pred_indices])))
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

###########################run GCCMshp#########################################################
library(parallel)
library(foreach)
library(doParallel)
library("spdep")  #  read shape file data

#######################################################################################
# GCCMshpmainfunction<-function(minlib,maxlib,intervalib,E,xName,yName,coordsX_Name,coordsY_Name,cores,formattedselectshppath,formattedSaveshpPath)
  
# GCCMshpmainfunction<-function(minlib,maxlib,intervalib,E,xName,yName,coordsX_Name,coordsY_Name,cores,formattedselectshppath,formattedSaveshpPath)
GCCMshpmainfunction<-function(minlib,maxlib,intervalib,E,xName,yName,coordsX_Name,coordsY_Name,cores,formattedselectshppath,formattedSaveshpPath)
{
  columbus <- st_read((formattedselectshppath)[1], quiet=TRUE)
  neighborMatrix <- poly2nb(as(columbus, "Spatial"))
  
  lib_sizes<-seq(minlib,maxlib,intervalib)
 
  n <- NROW(columbus)            # get the total mumber of spatial records
  lib <- c(1, n)                  # set the libray to confine the  spatial records jonint the construction of state spae 
  pred <- c(1, n)                 # set the spatial records which will be predicted based on the   state spae

  columbus<-as.data.frame(columbus)
  
  y<-columbus[,yName] 
  
  # coordsX<-columbus[,coordsX_Name]
  # 
  # coordsY<-columbus[,coordsY_Name]
  # 
  # coords<-columbus[,c(coordsX_Name,coordsY_Name)]
  # 
  # lmModel<-lm(as.formula(paste(yName,"~coordsX+coordsY",sep = "")),columbus ) # lmModeltest<-lm(as.formula(paste(yName,"~x_1+y_1",sep = "")),columbus )     #Remove the lieanr trend of effect variable  
  # 
  # prediction<-predict(lmModel,coords)
  # y<-y-prediction
  
  
  x<-columbus[,xName] 
  
  # lmModel2<-lm(as.formula(paste(xName,"~coordsX+coordsY",sep = "")),columbus )     #Remove the lieanr trend of cause variable  
  # prediction<-predict(lmModel2,coords)
  # x<-x-prediction
  lmModel<-lm(x ~ y, data.frame(x,y))
  prediction<-predict(lmModel,data.frame(y))
  x<-x-prediction
  
  embedings<-generateEmbedings(neighborMatrix,x,E)                       #generate the  embedings of cause variable  
  tau = 1
  b = E+2
 x_xmap_y <- GCCMLattice(embedings, y, lib_sizes, lib, pred, E,tau,b,cores) #predict y based on x  
  
  
  embedings<-generateEmbedings(neighborMatrix,y,E)                        #generate the  embedings of effect variable  
   y_xmap_x <- GCCMLattice(embedings, x, lib_sizes, lib, pred, E,tau,b,cores)  #predict x based on y  
  
  
  x_xmap_y$L <- as.factor(x_xmap_y$L)
  x_xmap_y_means <- do.call(rbind, lapply(split(x_xmap_y, x_xmap_y$L), function(x){max(0, mean(x$rho,na.rm=TRUE))}))
  #calculate the mean of prediction accuray, measure by Pearson correlation
  
  
  y_xmap_x$L <- as.factor(y_xmap_x$L)
  y_xmap_x_means <- do.call(rbind, lapply(split(y_xmap_x, y_xmap_x$L), function(x){max(0, mean(x$rho,na.rm=TRUE))}))
  #calculate the mean of prediction accuray, measure by Pearson correlation
  
  
  x_xmap_y_Sig<- significance(x_xmap_y_means,pred[2])    #Test the significance of the prediciton accuray
  y_xmap_x_Sig<- significance(y_xmap_x_means,pred[2])    #Test the significance of the prediciton accuray
  
  x_xmap_y_interval<- confidence(x_xmap_y_means,pred[2]) #calculate the  95%. confidence interval  of the prediciton accuray
  
  colnames(x_xmap_y_interval)<-c("x_xmap_y_upper","x_xmap_y_lower")
  
  y_xmap_x_interval<- confidence(y_xmap_x_means,pred[2])
  colnames(y_xmap_x_interval)<-c("y_xmap_x_upper","y_xmap_x_lower")  #calculate the  95%. confidence interval of the prediciton accuray
  
  # save(minlib,maxlib,intervalib,E,xName,yName,cores,lib_sizes,x_xmap_y_means,
  #      y_xmap_x_means,x_xmap_y_Sig,y_xmap_x_Sig,x_xmap_y_interval,y_xmap_x_interval,file = formattedSaveshpPath)
  
  variables <- list(
    xName=xName,
    yName=yName,
    # coordsX_Name=coordsX_Name,
    # coordsY_Name=coordsY_Name,
    minlib = minlib,
    maxlib = maxlib,
    intervalib = intervalib,
    E = E,
    tau=tau,
    b=b,
    cores = cores,
    lib_sizes = paste(lib_sizes, collapse = ", "),  # 如果是向量，转化为字符串
    x_xmap_y_means = paste(x_xmap_y_means, collapse = ", "),
    y_xmap_x_means = paste(y_xmap_x_means, collapse = ", "),
    x_xmap_y_Sig = paste(x_xmap_y_Sig, collapse = ", "),
    y_xmap_x_Sig = paste(y_xmap_x_Sig, collapse = ", "),
    x_xmap_y_interval = paste(x_xmap_y_interval, collapse = ", "),
    y_xmap_x_interval = paste(y_xmap_x_interval, collapse = ", ")
  )
  
  
  output_file <- formattedSaveshpPath
  sink(file(output_file, encoding = "UTF-8"))
  
  
  for (var_name in names(variables)) {
    cat(var_name, ":", variables[[var_name]], "\n")
  }
  
  sink()  # 
  
  
  
  return()
}










