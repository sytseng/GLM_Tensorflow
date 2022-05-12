#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import tensorflow as tf
from time import time
import matplotlib
import matplotlib.pyplot as plt
#np.seterr(invalid='ignore');


# In[2]:


def fit_glm(response_matrix, pred_matrix, pred_name_list, 
            loss_type = 'poisson', activation = 'exp', regularization = 'elastic_net', l1_ratio = 0.5, smooth_strength = 0., 
            lambda_series = 10.0 ** np.linspace(-1, -8, 30), learning_rate = 0.001,
            convergence_tol = 1e-9, max_iter_per_lambda = 10**4, min_iter_per_lambda = 10**2, 
            opt = 'adam', device = '/gpu:0', verbose = True):
    ''' Fit Poisson GLM
    Input:
    response_matrix: neural response data
    pred_matrix: design matrix
    pred_name_list: list of predictor names
    loss_type: loss type, e.g. poisson, gaussian, or binominal
    activation: activation function, e.g. exp, linear, or sigmoid
    reg: regularization type, elsatic_net or group_lasso
    l1_ratio: ratio for L1 regularization for elastic net
    smooth_ratio: ratio for smooth penalty
    lambda_series: list of lambda values for regularization
    
    Return: 
    w_series: list of weights [w0, w] of different lambdas
    lambda_series: list of lambda values for regularization
    '''
    
    n_roi = response_matrix.shape[1]
    n_t = response_matrix.shape[0]
    n_features = pred_matrix.shape[1]
    w_series = []
    loss_trace = np.array([])
    lambda_trace = np.array([])
    if len(pred_name_list) > 0:
        prior, grouping_mat, feature_group_size, _, _ = make_prior_matrix(pred_name_list)
    
    # model definition
    tf.reset_default_graph()
    with tf.device(device):
        # inputs
        X = tf.convert_to_tensor(pred_matrix, dtype=tf.float32)
        Y = tf.convert_to_tensor(response_matrix, dtype=tf.float32)

        if len(pred_name_list) > 0:
            P = tf.convert_to_tensor(prior, dtype=tf.float32)
            grouping_mat = tf.convert_to_tensor(grouping_mat, dtype=tf.float32)
            feature_group_size = tf.convert_to_tensor(feature_group_size, dtype=tf.float32)
        lam = tf.placeholder(tf.float32, name='lambda')

        # weights
        w0 = tf.Variable(tf.zeros((1, n_roi), dtype=tf.float32), name='intercept')
        initial_w = np.zeros((n_features, n_roi), np.float32) + 1e-5  # The gradient is not defined at zero so initialize with small number
        w = tf.Variable(tf.convert_to_tensor(initial_w), name='weight')

        # model definition
        Y_hat = tf.matmul(X, w) + w0
            
        if activation == 'exp':
            Y_act = tf.exp(Y_hat)
        
        # take relu
        elif activation == 'relu':
            Y_act = tf.nn.relu(Y_hat)
        
        # take softplus
        elif activation == 'softplus':
            Y_act = tf.math.softplus(Y_hat)
            
        # elastic net regularization
        if regularization== 'elastic_net':
            reg = lam * ((1. - l1_ratio) * tf.reduce_sum(tf.square(w)/2.0) + 
                                                       l1_ratio * tf.reduce_sum(tf.abs(w)))

        # group lasso penalty
        elif regularization == 'group_lasso':
            reg = lam * tf.reduce_sum(tf.sqrt(tf.matmul(grouping_mat, tf.square(w))) *
                                                                    tf.sqrt(feature_group_size)[:, None])
        
        # smoothness-encouraging regularization
        if smooth_strength > 0.:
            reg += smooth_strength * tf.einsum('ij,ik,kj->', w, P, w)

        # Loss function
        if np.logical_and(loss_type == 'poisson', activation == 'exp'):
            loss = tf.reduce_sum(tf.nn.log_poisson_loss(Y, Y_hat)) / n_t / n_roi + reg / n_roi
        
        elif np.logical_and(loss_type == 'poisson', activation != 'exp'):
            loss = tf.reduce_sum(Y_act - Y * tf.log(Y_act + 1e-33)) / n_t / n_roi + reg / n_roi
    
        elif loss_type == 'exponential':
            loss = tf.reduce_sum(Y/(Y_act + 1e-16) + tf.log(Y_act)) / n_t / n_roi + reg / n_roi
            
        elif np.logical_and(loss_type == 'gaussian', activation == 'linear'):
            loss = tf.reduce_sum(tf.square(Y - Y_hat)) / n_t / n_roi + reg / n_roi
            
        elif np.logical_and(loss_type == 'binominal', activation == 'sigmoid'):
            loss = tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(labels=Y, logits=Y_hat)) / n_t / n_roi + reg / n_roi
            
        # compute average null deviance
        null_dev = np.full((n_roi,),np.NaN)
        for ii in range(n_roi):
            this_Y = response_matrix[:,ii]
            null_dev[ii]= null_deviance(this_Y, loss_type = loss_type)

        avg_dev = np.sum(null_dev)/n_t/n_roi

        # Optimizer:
        non_optimizer_vars = set(tf.global_variables())
        if opt =='adam':
            optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)

        elif opt =='sgdm':
            optimizer = tf.train.MomentumOptimizer(learning_rate=learning_rate, momentum=0.9)
        
        # break up minimize into compute_gradients and apply_gradients
#         train_op = optimizer.minimize(loss)
        grads_and_vars = optimizer.compute_gradients(loss)
        wGrads = grads_and_vars[1][0]
        wGrads_clean = tf.where(tf.is_nan(wGrads), tf.zeros_like(wGrads), wGrads)
        train_op = optimizer.apply_gradients([grads_and_vars[0], [wGrads_clean, grads_and_vars[1][1]]])
        optimizer_vars = set(tf.global_variables()) - non_optimizer_vars
        
        # gpu setting:
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.5)
        config = tf.ConfigProto(gpu_options=gpu_options)
        
        # set up initializers
        allVarInitializer = tf.global_variables_initializer()
        optimizerInitializer = tf.variables_initializer(optimizer_vars)
        
        # fit model:
        lambda_index = 0
        iter_this_lambda = 0
        with tf.Session(config=config) as sess:
            start_time = time()
            # initialize all variables
            sess.run(allVarInitializer)
            
            while True:
                iter_this_lambda += 1
                feed_dict = {lam: lambda_series[lambda_index]}

                # Train:
                loss_this_iter = sess.run([train_op, loss], feed_dict=feed_dict)[1]
                assert (not np.isnan(loss_this_iter)), 'Loss is nan -- check.'
                loss_trace = np.append(loss_trace, loss_this_iter)
                lambda_trace = np.append(lambda_trace, feed_dict[lam])

                # Check for convergence:
                if (iter_this_lambda % 100) == 0 and (iter_this_lambda > min_iter_per_lambda):
                    loss_diff = np.mean(-np.diff(loss_trace[-min_iter_per_lambda:])) # Mean over last X steps.

                    if (np.absolute(loss_diff) < convergence_tol*avg_dev/learning_rate) or (iter_this_lambda == max_iter_per_lambda):
                        if verbose:
                            if iter_this_lambda == max_iter_per_lambda:
                                print('Fitting with Lambda {} iter {} did not converge (loss diff = {:1.8f}).'
                                      .format(lambda_index, iter_this_lambda, loss_diff))
                            else:
                                print('Fitting with Lambda {} iter {} converged (loss diff = {:1.8f}).'
                                      .format(lambda_index, iter_this_lambda, loss_diff))
                        w_series.append(sess.run([w0, w], feed_dict=feed_dict))  

                        lambda_index += 1
                        iter_this_lambda = 0

                        # Re-initialize ADAM optimizer moments:
                        sess.run(optimizerInitializer)

                        if lambda_index == lambda_series.size:
                            if verbose: print('Finished lambda series.')
                            break
                    else:
                        if verbose: print('Lambda {} loss: {:1.5f} diff: {:1.7f}'.format(lambda_index, loss_this_iter, loss_diff))

        if verbose: print('Fitting took {:1.1f} seconds.'.format(time() - start_time))
        
        return w_series, lambda_series, loss_trace, lambda_trace


# In[3]:


def fit_glm_cv(response_matrix, pred_matrix, pred_name_list, n_folds, train_ind, test_ind,
               loss_type = 'poisson', activation = 'exp', regularization = 'elastic_net', l1_ratio = 0.5, smooth_strength = 0., 
               lambda_series = 10.0 ** np.linspace(-1, -8, 30), learning_rate = 0.001,
               convergence_tol = 1e-10, max_iter_per_lambda = 10**4, min_iter_per_lambda = 10**2, 
               opt = 'adam', device = '/gpu:0', verbose = True):
    
    ''' Fit GLM with cross validation
    Input:
    response_matrix: neural response data
    pred_matrix: design matrix
    pred_name_list: list of predictor names
    n_folds: number of CV fold
    train_ind: dictionary with training frame indices for each fold and the all training frames at last "fold"
    test_ind: dictionary with test frame indices for each fold (basically for "validation")
    loss_type: loss type, e.g. poisson, gaussian, or binominal
    activation: activation function, e.g. exp, linear, or sigmoid
    regularization: regularization type, elsatic_net or group_lasso
    l1_ratio: ratio for L1 regularization for elastic net
    smooth_strength: strength for smooth penalty
    lambda_series: list of lambda values for regularization
    learning_rate: learning rate for the optimizer
    
    Return: 
    w_series_dict: dictionary for each CV fold with list of weights [w0, w] of different lambdas
    lambda_series: list of lambda values for regularization
    loss_trace_dict: loss trace for each fold (set as empty now)
    lambda_trace_dict: lambda trace for each fold (set as empty now)
    all_prediction: prediction for the held-out data
    all_deviance: deviance for the held-out data
    '''

    n_roi = response_matrix.shape[1]
    n_features = pred_matrix.shape[1]
    w_series_dict = {}
    loss_trace_dict = {}
    lambda_trace_dict = {}
    if len(pred_name_list) > 0:
        prior, grouping_mat, feature_group_size, _, _ = make_prior_matrix(pred_name_list)
    
    # model definition
    tf.reset_default_graph()
    with tf.device(device):
        # inputs   
        n_t = tf.Variable(response_matrix.shape[0], dtype=tf.float32, trainable=False)
        X_const = tf.constant(pred_matrix, dtype=tf.float32)
        Y_const = tf.constant(response_matrix, dtype=tf.float32)
        frame_ind = tf.placeholder(tf.int32, shape = (None), name = 'frame_ind')
        X = tf.gather(X_const , frame_ind)
        Y = tf.gather(Y_const , frame_ind)
        if len(pred_name_list) > 0:
            P = tf.convert_to_tensor(prior, dtype=tf.float32)
            grouping_mat = tf.convert_to_tensor(grouping_mat, dtype=tf.float32)
            feature_group_size = tf.convert_to_tensor(feature_group_size, dtype=tf.float32)
        lam = tf.placeholder(tf.float32, name='lambda')

        # weights
        w0 = tf.Variable(tf.zeros((1, n_roi), dtype=tf.float32), name='intercept')
        initial_w = np.zeros((n_features, n_roi), np.float32) + 1e-5  # The gradient is not defined at zero so initialize with small number
        w = tf.Variable(initial_w, name='weight')

        # model definition
        Y_hat = tf.matmul(X, w) + w0
            
        if activation == 'exp':
            Y_act = tf.exp(Y_hat)
        
        # take relu
        elif activation == 'relu':
            Y_act = tf.nn.relu(Y_hat)
        
        # take softplus
        elif activation == 'softplus':
            Y_act = tf.math.softplus(Y_hat)
            
        # elastic net regularization
        if regularization== 'elastic_net':
            reg = lam * ((1. - l1_ratio) * tf.reduce_sum(tf.square(w)/2.0) + 
                                                       l1_ratio * tf.reduce_sum(tf.abs(w)))

        # group lasso penalty
        elif regularization == 'group_lasso':
            reg = lam * tf.reduce_sum(tf.sqrt(tf.matmul(grouping_mat, tf.square(w))) *
                                                                    tf.sqrt(feature_group_size)[:, None])
        
        # smoothness-encouraging regularization
        if smooth_strength > 0.:
            reg += smooth_strength * tf.einsum('ij,ik,kj->', w, P, w)

        # Loss function
        if np.logical_and(loss_type == 'poisson', activation == 'exp'):
            pois_loss = tf.nn.log_poisson_loss(Y, Y_hat)
            loss = tf.reduce_sum(pois_loss) / n_t / n_roi + reg / n_roi
        
        elif np.logical_and(loss_type == 'poisson', activation != 'exp'):
            loss = tf.reduce_sum(Y_act - Y * tf.log(Y_act + 1e-33)) / n_t / n_roi + reg / n_roi
               
        elif loss_type == 'exponential':
            loss = tf.reduce_sum(Y/(Y_act + 1e-16) + tf.log(Y_act)) / n_t / n_roi + reg / n_roi
            
        elif np.logical_and(loss_type == 'gaussian', activation == 'linear'):
            loss = tf.reduce_sum(tf.square(Y - Y_hat)) / n_t / n_roi + reg / n_roi
            
        elif np.logical_and(loss_type == 'binominal', activation == 'sigmoid'):
            loss = tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(labels=Y, logits=Y_hat)) / n_t / n_roi + reg / n_roi

            
        # Optimizer:
        non_optimizer_vars = set(tf.global_variables())
        if opt =='adam':
            optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)

        elif opt =='sgdm':
            optimizer = tf.train.MomentumOptimizer(learning_rate=learning_rate, momentum=0.9)
        
        # break minimize into compute_gradients and apply_gradients
#         train_op = optimizer.minimize(loss)
        grads_and_vars = optimizer.compute_gradients(loss)
        wGrads = grads_and_vars[1][0]
        wGrads_clean = tf.where(tf.is_nan(wGrads), tf.zeros_like(wGrads), wGrads) # replace nan's with 0's
        train_op = optimizer.apply_gradients([grads_and_vars[0], [wGrads_clean, grads_and_vars[1][1]]])
        optimizer_vars = set(tf.global_variables()) - non_optimizer_vars
        
        # gpu setting:
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.5)
        config = tf.ConfigProto(gpu_options=gpu_options)
        
        # compute average null deviance
        null_dev = np.full((n_roi,),np.NaN)
        for ii in range(n_roi):
            this_Y = response_matrix[:,ii]
            null_dev[ii]= null_deviance(this_Y, loss_type = loss_type)

        avg_dev = np.sum(null_dev)/response_matrix.shape[0]/n_roi
        
        # prelocate
        all_prediction = [np.full(response_matrix.shape,np.NaN) for idx, _ in enumerate(lambda_series)]
        all_deviance = [np.full((n_folds,n_roi),np.NaN) for idx, _ in enumerate(lambda_series)]
        
        # set up initializers
        allVarInitializer = tf.global_variables_initializer()
        optimizerInitializer = tf.variables_initializer(optimizer_vars)
        
        # fit model
        with tf.Session(config=config) as sess:
            start_time = time()
            for n_fold in range(n_folds+1):
                print('nFold =', n_fold)
                lambda_index = 0
                iter_this_lambda = 0
                w_series = []

                sess.run(allVarInitializer)
                feed_dict = {lam: lambda_series[lambda_index], frame_ind: train_ind[n_fold]}
                
                # re-assign n_t
                assign_n_t = tf.assign(n_t, train_ind[n_fold].shape[0])
                sess.run(assign_n_t)
                
                # get values for 1st loss
                prevLoss = sess.run(loss, feed_dict=feed_dict)

                while True:
                    iter_this_lambda += min_iter_per_lambda

                    # Train:
                    for i in range(min_iter_per_lambda):
                        _ = sess.run(train_op, feed_dict=feed_dict)
                    newLoss = sess.run(loss, feed_dict=feed_dict)                    
                    assert (not np.isnan(newLoss)), 'Loss is nan -- check.'
                    lossDiff = (prevLoss-newLoss)/min_iter_per_lambda                                       
                    prevLoss = newLoss


                    # Check for convergence:
                    #if (iter_this_lambda % 100) == 0 and (iter_this_lambda > min_iter_per_lambda):
                    #    loss_diff = np.mean(-np.diff(loss_trace[-min_iter_per_lambda:])) # Mean over last X steps.
                    if ((iter_this_lambda > min_iter_per_lambda) and (np.abs(lossDiff)<convergence_tol*avg_dev/learning_rate)) or (iter_this_lambda >= max_iter_per_lambda) :
                        if verbose: 
                            if iter_this_lambda >= max_iter_per_lambda:
                                print('Fitting with Lambda {} iter {} did not converge (loss diff = {:1.8f}).'
                                      .format(lambda_index, iter_this_lambda, lossDiff))
                            else:
                                print('Fitting with Lambda {} iter {} converged (loss diff = {:1.8f}).'
                                      .format(lambda_index, iter_this_lambda, lossDiff))
                        w_series.append(sess.run([w0, w], feed_dict=feed_dict))  

                        lambda_index += 1
                        iter_this_lambda = 0

                        # Re-initialize ADAM optimizer moments:
                        sess.run(optimizerInitializer)

                        if lambda_index == lambda_series.size:
                            if verbose: print('Finished lambda series.')
                            break
                        else:
                            feed_dict = {lam: lambda_series[lambda_index], frame_ind: train_ind[n_fold]}
                                
                    else:
                        if verbose: print('Lambda {} loss: {:1.5f} diff: {:1.7f}'.format(lambda_index, newLoss, lossDiff))

                w_series_dict[n_fold] = w_series
                
                if n_fold < n_folds:
                    testFrames = test_ind[n_fold]
                    X_test = pred_matrix[testFrames,:]
                    Y_test = response_matrix[testFrames,:]
                    for this_idx, this_w in enumerate(w_series):
                        prediction = make_prediction(X_test, this_w[1], this_w[0], activation = activation)
                        _, d_model, _ = deviance(prediction, Y_test, loss_type = loss_type)
                        all_prediction[this_idx][testFrames,:] = prediction
                        all_deviance[this_idx][n_fold,:] = d_model.reshape(1,-1)
                
        if verbose: print('Fitting took {:1.1f} seconds.'.format(time() - start_time))
        
        return w_series_dict, lambda_series, loss_trace_dict, lambda_trace_dict, all_prediction, all_deviance


# In[4]:


def stable(x):
    return x + 10 * np.finfo(x.dtype).tiny


# In[5]:


def deviance(mu, y, loss_type = 'poisson'):
    """Calculate Poisson-devinace-explained between pairs of columns from the matrices mu and y.
    See MATLAB getDeviance.m
    The version here has improved numerical stability.
    """
    assert (mu.shape == y.shape), "Shapes " + str(mu.shape) + " and " + str(y.shape) + " don't match!"

    mean_y = np.mean(y, axis=0)
    log_y = np.log(stable(y))

    if loss_type == 'poisson':
        d_model = 2.0 * np.sum(y * (log_y - np.log(stable(mu))) + mu - y, axis=0)
        d_null = 2.0 * np.sum(y * (log_y - np.log(stable(mean_y))) + mean_y - y, axis=0)
    elif loss_type == 'gaussian':
        d_model = np.sum((y - mu)**2, axis=0)
        d_null = np.sum((y - mean_y)**2, axis=0)
    elif loss_type == 'exponential':
        d_model = 2.0 * np.sum(np.log(stable(mu)) - log_y + y * (y - mu), axis=0)
        d_null = 2.0 * np.sum(np.log(stable(mean_y)) - log_y + y * (y - mean_y), axis=0)
    elif loss_type == 'binominal':
        d_model = 2.0 * np.sum(-y*np.log(stable(mu))-(1.-y)*np.log(stable(1.-mu))
                               +y*np.log(stable(y))+(1.-y)*np.log(stable(1.-y)), axis=0)
        d_null = 2.0 * np.sum(-y*np.log(stable(mean_y))-(1.-y)*np.log(stable(1.-mean_y))
                               +y*np.log(stable(y))+(1.-y)*np.log(stable(1.-y)), axis=0)

    dev = 1.0 - d_model/stable(d_null)    
    if isinstance(dev, type(y)): # if dev is still an ndarray (skip if is a single number)
        dev[mean_y == 0] = 0  # If mean_y == 0, we get 0 for model and null deviance, i.e. 0/0 in the deviance fraction.
    return dev, d_model, d_null


# In[6]:


def null_deviance(y, loss_type):
    mean_y = np.mean(y, axis=0)
    log_y = np.log(stable(y))
    if loss_type == 'poisson':
        d_null = 2.0 * np.sum(y * (log_y - np.log(stable(mean_y))) + mean_y - y, axis=0)
    elif loss_type == 'exponential':
        d_null = 2.0 * np.sum(np.log(stable(mean_y)) - log_y + y * (y - mean_y), axis=0)
    elif loss_type == 'gaussian':
        d_null = np.sum((y - mean_y)**2, axis=0)
    elif loss_type == 'binominal':
        d_null = 2.0 * np.sum(-y*np.log(stable(mean_y))-(1.-y)*np.log(stable(1.-mean_y))
                               +y*np.log(stable(y))+(1.-y)*np.log(stable(1.-y)), axis=0)
    return d_null


# In[7]:


def make_prediction(X, w, bias, activation = 'exp'):
    if activation == 'exp':
        prediction = np.exp(bias + np.matmul(X, w))
    elif activation == 'relu':
        prediction = np.maximum((bias + np.matmul(X, w)),0) 
    elif activation == 'softplus':
        prediction = np.log(stable(np.exp(bias + np.matmul(X, w)) + 1.)) # take softplus = log(exp(features) + 1
    elif activation == 'linear':
        prediction = bias + np.matmul(X, w)
    elif activation == 'sigmoid':
        prediction = 1./(1.+np.exp(-bias - np.matmul(X, w)))
    return prediction


# In[8]:


def make_prediction_cv(X, n_sources, n_folds, val_ind, w_series_dict, all_lambda_ind, activation = 'exp'):

    pred = np.empty((X.shape[0], n_sources))
    
    # loop over CV folds for making prediction on validation data
    for n_fold in range(n_folds):
        # grab validation indices for this fold
        these_val_frames = val_ind[n_fold]
        
        # grab w and w0 for this fold
        this_w0_cv = []
        this_w_cv = []
        for n_source in range(n_sources):
            this_lambda_ind = all_lambda_ind[n_source]
            w0 = w_series_dict[n_fold][this_lambda_ind][0][:,n_source]
            w = w_series_dict[n_fold][this_lambda_ind][1][:,n_source]
            this_w0_cv.append(w0)
            this_w_cv.append(w)
                
        this_w_cv = np.stack(this_w_cv, axis = 1)
        this_w0_cv = np.stack(this_w0_cv, axis = 1)
        
        # make predictions on validation frames of this fold
        pred[these_val_frames,:] = make_prediction(X[these_val_frames,:], this_w_cv, this_w0_cv, activation = activation)
    
    return pred


# In[9]:


def calculate_fit_quality(w_series, lambda_series, X, Y, loss_type = 'poisson', activation = 'exp', make_fig = True):
    '''Make prediction and calculate fit quality (fraction deviance explained)'''
    # compute validation loss for all lambdas
    all_fit_qual = []
    all_d_model = []
    all_d_null = []
    all_prediction = []

    for idx, w in enumerate(w_series):
        # make predictions
        prediction = make_prediction(X, w[1], w[0], activation = activation)
        all_prediction.append(prediction)
        # calculate fraction deviance explained, model deviance, and null deviance
        dev, d_model, d_null = deviance(prediction, Y, loss_type = loss_type)
        all_d_model.append(d_model)
        all_fit_qual.append(dev)
        if idx == 0:
            all_d_null = d_null
    all_fit_qual = np.stack(all_fit_qual, axis=0)
    all_d_model = np.stack(all_d_model, axis=0)

    # plot fraction deviance explained vs. lambda (for some traces)
    if make_fig:
        fig, ax = plt.subplots(1,1)
        ax.plot(np.log10(lambda_series), all_fit_qual, color='k', linewidth=0.5)
        ax.set_ylim((-0.1,1))
        ax.set_xlabel('log_lambda')
        ax.set_ylabel('Fraction deviance explained')
        ax.set_title('Fraction deviance explained vs. lambda')
        
    return all_fit_qual, all_d_model, all_d_null, all_prediction


# In[10]:


def calculate_fit_quality_cv(lambda_series, all_prediction, response_matrix,
                             loss_type = 'poisson', activation = 'exp', make_fig = True):
    all_fit_qual = []
    all_d_model = []

    for idx, _ in enumerate(lambda_series):
        prediction = all_prediction[idx]
        dev, d_model, d_null = deviance(prediction, response_matrix, loss_type = loss_type)
        all_d_model.append(d_model)
        all_fit_qual.append(dev)
        if idx == 0:
            all_d_null = d_null
    all_fit_qual = np.stack(all_fit_qual, axis=0)
    all_d_model = np.stack(all_d_model, axis=0)

    # plot fraction deviance explained vs. lambda (for some traces)
    if make_fig:
        fig, ax = plt.subplots(1,1)
        ax.plot(np.log10(lambda_series), all_fit_qual, color='k', linewidth=0.5)
        ax.set_ylim((-0.1,1))
        ax.set_xlabel('log_lambda')
        ax.set_ylabel('Fraction deviance explained')
        ax.set_title('Fraction deviance explained vs. lambda')
        
    return all_fit_qual, all_d_model, all_d_null


# In[11]:


def select_model(w_series, lambda_series, X_val, Y_val, X_test, Y_test, 
                 min_lambda = 1e-4, loss_type = 'poisson', activation = 'exp', make_fig = True):
    '''Select model with the highest fraction deviance explained (or with some small regulairzation)'''
    
    # calculate fit quality (frac deviance explained) using validation set
    all_fit_qual, _, _, _ = calculate_fit_quality(w_series, lambda_series, X_val, Y_val, 
                                                  loss_type = loss_type, make_fig = False)
    
    # select best model for each source
    all_w0 = []
    all_w = []
    all_lambda = []
    all_lambda_ind = []
    all_dev = []
    all_best_d_model = []
    all_best_d_null = []
    all_best_dev_expl = []
    for idx in range(Y_val.shape[1]):
        # find lambda for highest fraction deviance explained (or with some small regulairzation)
        if min_lambda > np.finfo(np.array([0.]).dtype).tiny:            
            best_lambda_ind = np.min([np.argmax(all_fit_qual[:,idx]), 
                                      np.argwhere(lambda_series<min_lambda)[0][0]-1])
        else:
            best_lambda_ind = np.argmax(all_fit_qual[:,idx])

        best_lambda = lambda_series[best_lambda_ind]    
        best_w0 = w_series[best_lambda_ind][0][:,idx]
        best_w = w_series[best_lambda_ind][1][:,idx]
        
        # make prediction on test set and calculate fraction deviance explained, model deviance, and null deviance
        prediction = make_prediction(X_test, best_w, best_w0, activation = activation)
        best_frac_deviance, best_d_model, best_d_null = deviance(prediction, Y_test[:,idx], loss_type = loss_type)
        best_dev_expl = best_d_null - best_d_model
        
        all_w0.append(best_w0)
        all_w.append(best_w)
        all_dev.append(best_frac_deviance)
        all_lambda.append(best_lambda)
        all_lambda_ind.append(best_lambda_ind)
        all_best_d_model.append(best_d_model)
        all_best_d_null.append(best_d_null)
        all_best_dev_expl.append(best_dev_expl)

    if make_fig:
        # plot CDF for fraction deviance explained / scatter of  deviance explained vs null deviance
        from statsmodels.distributions.empirical_distribution import ECDF
        ecdf = ECDF(all_dev)
        x = np.arange(0,1,0.01)
        this_ecdf = ecdf(x)
        print('Mean deviance explained =', np.mean(all_dev))

        fig,axes = plt.subplots(1,2, figsize = (8,3))
        axes[0].plot(x,this_ecdf)    
        axes[0].set_xlabel('Fraction deviance explained')
        axes[0].set_ylabel('Cumulative density')

        axes[1].plot(all_best_d_null, all_best_dev_expl,'.',markersize = 3)
        axes[1].plot(np.linspace(0,np.max(all_best_d_null),100), np.linspace(0,np.max(all_best_d_null),100),
                     linestyle='--', linewidth = 1,color = (0.5,0.5,0.5))
        axes[1].set_xlim([0, np.max(all_best_d_null)])
        axes[1].set_ylim([0, np.max(all_best_dev_expl)])
        axes[1].set_xlabel('Deviance for null model')
        axes[1].set_ylabel('Deviance explained')
        plt.tight_layout()
        
    return all_dev, all_w, all_w0, all_lambda, all_lambda_ind


# In[12]:


def select_model_cv(w_series_dict, lambda_series, all_deviance, n_folds, se_fraction, 
                    all_fit_qual, all_d_model = list(), all_d_null = list(), make_fig = True):
    
    n_sources = all_deviance[0].shape[1]
    
    # compute average deviance and standard error
    avg_deviance = [np.mean(dev, axis = 0) for dev in all_deviance]
    avg_deviance = np.stack(avg_deviance, axis = 0)
    se_deviance = [np.std(dev, axis = 0)/np.sqrt(n_folds) for dev in all_deviance]
    se_deviance = np.stack(se_deviance, axis = 0)
#     sd_deviance = [np.std(dev, axis = 0) for dev in all_deviance]
#     sd_deviance = np.stack(sd_deviance, axis = 0)
    
    # initiate
    all_w0 = []
    all_w = []
    all_lambda = []
    all_dev = []
    all_min_lambda_ind = []
    all_lambda_ind = []
    all_min_lambda = []
    all_selected_dev_expl = []

    for idx in range(n_sources):
        min_deviance = np.min(avg_deviance[:,idx])
        min_dev_lambda_ind = np.argmin(avg_deviance[:,idx])
        this_se = se_deviance[min_dev_lambda_ind,idx]
        threshold = min_deviance + this_se * se_fraction
#         this_sd = sd_deviance[min_dev_lambda_ind,idx]
#         threshold = min_deviance + this_sd * sd_fraction

        # find the lambda index with avg deviance smaller than threshold
        this_lambda_ind = np.argwhere(avg_deviance[:,idx] <= threshold)[0][0]
        this_lambda = lambda_series[this_lambda_ind]
        
        # find fraction deviance explained for the selected lambda
        this_frac_dev = all_fit_qual[this_lambda_ind,idx]
        
        # find the corresponding weights for the lambda 
        # note that w_series_dict[n_folds] returns the weights fitted with full data
        this_w0 = w_series_dict[n_folds][this_lambda_ind][0][:,idx]
        this_w = w_series_dict[n_folds][this_lambda_ind][1][:,idx]

        # collect all parameters
        all_lambda_ind.append(this_lambda_ind)
        all_lambda.append(this_lambda) 
        all_min_lambda_ind.append(min_dev_lambda_ind)
        all_min_lambda.append(lambda_series[min_dev_lambda_ind])
        all_w0.append(this_w0)
        all_w.append(this_w)
        all_dev.append(this_frac_dev) 
        
        # collect deviance explained if we want to make plot
        if make_fig:
            this_d_model = all_d_model[this_lambda_ind,idx]
            all_selected_dev_expl.append(all_d_null[idx] - this_d_model)

    # plot CDF for fraction deviance explained / scatter of deviance explained vs null deviance
    if make_fig:
        from statsmodels.distributions.empirical_distribution import ECDF
        ecdf = ECDF(all_dev)
        x = np.arange(0,1,0.01)
        this_ecdf = ecdf(x)
        print('Mean deviance explained =', np.mean(all_dev))

        fig,axes = plt.subplots(1,2, figsize = (8,3))
        axes[0].plot(x,this_ecdf)    
        axes[0].set_xlabel('Fraction deviance explained')
        axes[0].set_ylabel('Cumulative density')

        axes[1].plot(all_d_null, all_selected_dev_expl,'.',markersize = 3)
        axes[1].plot(np.linspace(0,np.max(all_d_null),100), np.linspace(0,np.max(all_d_null),100),
                     linestyle='--', linewidth = 1,color = (0.5,0.5,0.5))
        axes[1].set_xlim([0, np.max(all_d_null)])
        axes[1].set_ylim([0, np.max(all_selected_dev_expl)])
        axes[1].set_xlabel('Deviance for null model')
        axes[1].set_ylabel('Deviance explained')
        plt.tight_layout()
        
    return all_dev, all_w, all_w0, all_lambda, all_lambda_ind

