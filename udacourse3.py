#import libraries
import numpy as np
import pandas as pd
import scipy.stats as stats
import udacourse3 #my library
import progressbar
import pickle
#import tests as t
#import helper as h

#graphs
import seaborn as sns
import matplotlib.patches as mpatches
import matplotlib.style as mstyles
import matplotlib.pyplot as mpyplots
#% matplotlib inline
#from matplotlib.pyplot import hist
#from matplotlib.figure import Figure

#First part
from statsmodels.stats import proportion as proptests
from statsmodels.stats.power import NormalIndPower
from statsmodels.stats.proportion import proportion_effectsize
from time import time

#second part
from scipy.stats import spearmanr
from scipy.stats import kendalltau
from scipy.sparse import csr_matrix
from collections import defaultdict
from IPython.display import HTML

###Recomendation Engines - Collaborative Filter#################################
#########1#########2#########3#########4#########5#########6#########7#########8
def fn_all_recommendation_collab(df_dist,
                                 df_user_movie,
                                 df_movie,
                                 num_rec=10,
                                 limit=100,
                                 min_rating=7,
                                 sort=False, 
                                 verbose=False):
    '''This function creates a dictionnary for possible recommendations for
    a set of users, preprocessed at df_dist dataset.
    
    Source: Udacity Data Science Course - Lesson 6  - Identifying Reccomendations
    Forth Notebook - Class 17 - Collaborative Filtering -Recommendations with 
    MovieTweetings - Collaborative Filtering
    
    Input:
      - df_dist (mandatory) - a preprocessed dataset wit the euclidian distancies
        between two users (Pandas dataset)
      - df_user_movie (mandatory) - dataset in the shape user by movie -
        (Pandas dataset)
      - df_movie (mandatory) - dataset in the shape for movies - 
        (Pandas dataset)
      - num_rec (optional) - (int) number of recommended movies to return
      - limit (optional) - extra parameter for fn_find_closest_neighbor - 
        it limits the number of neighbors (normally 100 is more than enough) 
      - min_rating (optional) - extra parameter for fn_movie_liked2() - it is
        the worst score for considering a movie as liked (normally rate 7 is 
        enough)
      - sort (optional) - extra parameter for fn_movie_liked2() - if you want
        to show the best rated movies first (in this algorithm it is useless)
      - verbose (optional) - if you want some verbosity in your function -
        (Boolean, default=False)
    Output:
      - all_recs - a dictionary where each key is a user_id and the value is 
        an array of recommended movie titles
    '''
    if verbose:
        print('###function all_recommendations started')

    begin = time()

    #take all the unique users from df_dist dataset
    user_all = np.unique(df_dist['user1'])
    n_user_all = len(user_all)
    
    if verbose:
        print('*taken {} users to find recommendations'.format(n_user_all))
    
    #create a dictionnary for keeping all recommendations for each user
    all_rec = dict()
    
    #iterate users calling the function for recommendations
    for user in user_all:
        if verbose:
            print('*making recommendations for user', user)
            
        filt_dist=df_dist[df_dist['user1'] == user]
        all_rec[user] = udacourse3.fn_make_recommendation_collab(
                            filt_dist=filt_dist,
                            df_user_movie=df_user_movie,
                            df_movie=df_movie,
                            num_rec=num_rec,
                            limit=limit,
                            min_rating=min_rating,
                            sort=sort,
                            verbose=verbose)       
    end = time()
    
    if verbose:
        print('elapsed time: {:.4f}s'.format(end-begin))
    
    return all_rec

#########1#########2#########3#########4#########5#########6#########7#########8
def fn_calculate_distance(x, 
                          y,
                          dist_type=None,
                          verbose=False):
    '''This function calculates the euclidian distance between two points,
    considering that X and Y are gived as fractions of distance.
    
    Source: Udacity Data Science Course - Lesson 6 - Recomendation Engines - 
    Class 14 - Third Notebook - More Personalized Ways - Collaborative Filtering
    & Content Based - Measuring Similarity

    Inputs:
      - x (mandatory) - an array of matching length to array y
      - y (mandatory - an array of matching length to array x
      - dist_type (mandatory) - if none is informed, returns False and nothing
        is calculated:
        * 'euclidean' - calculates the euclidean distance (you pass through
          squares or edifices, as Superman)
        * 'manhattan' - calculates the manhattan distance (so you need to turn
          squares and yoy are not Superman!)
      - verbose (optional) - if you want some verbosity in your function -
        (Boolean, default=False)
    Output:
      - distance - the euclidean distance between X and Y.
    '''
    if verbose:
        print('###function calculate distances started')
        
    begin = time()
    
    if dist_type == None:
        if verbose:
            print('nothing was calculated, type was not informed')
        return False
        
    elif dist_type == 'euclidean':
        if verbose:
            print('calculating euclidian distance')
        distance = np.linalg.norm(x - y)
        
    elif dist_type == 'manhattan':
        if verbose:
            print('calculating manhattan distance')
        distance = sum(abs(e - s) for s, e in zip(x, y))
    
    end = time()
    
    if verbose:
        print('{} distance: {}'.format(dist_type, distance))
        print('elapsed time: {:.4f}s'.format(end-begin))
    
    return distance

#########1#########2#########3#########4#########5#########6#########7#########8
def fn_compute_correlation(x, 
                           y,
                           corr_type=None,
                           verbose=False):
    '''This function calculates correlation between two variables  A negative 
    value means an inverse correlation. Nearer to 1 or -1, stronger is the 
    correlation. More  data you have, lower tends to be the error. Two points 
    give us a 1 or -1 value.
    No correlation = 0 (a straight line). p-value is important too. A small
    p-value means greater confidence in our guess.
    
    Source: Udacity Data Science Course - Lesson 6 - Recomendation Engines - 
    Class 14 - Third Notebook - More Personalized Ways - Collaborative Filtering
    & Content Based - Measuring Similarity

    Inputs:
      - x (mandatory) - an array of matching length to array y (numpy Array)
      - y (mandatory) - an array of matching length to array x (numpy Array)
      - corr_ype (mandatory) - {'kendall_tau', 'pearson', 'spearman'}
        (default: None)
        * 'kendall_tau' - Kendall´s Tau correlation coefficient. It uses a more
          advancet technique and offers less variability on larger datasets.
          It is not so computional efficient, as it runs on O(n^2).
        * 'pearson' - is the most basic correlation coefficient. The denominator
          just normalizes the raw coorelation value. Even for quadratic it keeps
          suggesting a relationship.
        * 'spearman' - Spearman Correlation can deal easily with outliers, as 
          it uses ranked data. It don´t follow normal distribution, or binomial, 
          and is an example of non-parametric function. Runs on basis O(nLog(n)).
          from Udacity course notes:
          "Spearman's correlation can have perfect relationships e.g.:(-1, 1) 
          that aren't linear. Even for quadratic it keeps
          suggesting a relationship.
      - verbose (optional) - if you want some verbosity in your function -
        (Boolean, default=False)
    Output
      - correlation_spearman - the Spearman correlation coefficient for comparing 
        x and y.values from (-1 to zero to +1).        
    '''    
    if verbose:
        print('###function correlation started')
        
    begin = time()
    
    if corr_type is None:
        if verbose:
            print('no parameter calculated, you need to inform type')
        return False
    
    elif corr_type == 'intersection':
        #transform series into datafames
        df1 = x.to_frame()
        df2 = y.to_frame()
        correlation = pd.concat([df1, df2], axis=1).corr().iloc[0,1]
    
    elif corr_type == 'kendall_tau':
        #rank both data vectors
        x = x.rank()
        y = y.rank()        
        correlation = fn_onsquare(x, y, verbose=verbose)
    
    elif corr_type == 'pearson':
        correlation = fn_ologn(x, y, verbose=verbose)
        
    elif corr_type == 'spearman': 
        #rank both data vectors
        x = x.rank()
        y = y.rank()
        correlation = fn_ologn(x, y, verbose=verbose)
        
    else:
        if verbose:
            print('invalid parameter')
        return False
            
    end = time()
    
    if verbose:
        print('{} correlation: {:.4f}'.format(corr_type, correlation))
        print('elapsed time: {:4f}s'.format(end-begin))
                            
    return correlation 

#########1#########2#########3#########4#########5#########6#########7#########8
def fn_create_ranked_df(movie, 
                        review,
                        verbose=True):
    '''This function creates a ranked movies dataframe, that are sorted by 
    highest avg rating, more reviews, then time, and must have more than 4 
    ratings. Laterly this function can be forked for other purposes.
    
    Source: Udacity Data Science Course - Lesson 6 - Recomendation Engines - 
    Class 5 - First Notebook - Intro to Recommendation data - Part I - Finding
    Most Popular Movies.
    
    Inputs:
      - movie (mandatory) - the movies dataframe
      - review (mandatory) - the reviews dataframe
      - verbose (optional) - if you want some verbosity in your function -
        (Boolean, default=False)
    Output:
      - ranked_movie - a dataframe with movies 
    '''
    if verbose:
        print('###function create_ranked_df started')
        
    begin = time()
    
    #for review - take mean, count, last rating
    avg_rating = review.groupby('movie_id')['rating'].mean()
    num_rating = review.groupby('movie_id')['rating'].count()
    last_rating = pd.DataFrame(review.groupby('movie_id').max()['date'])
    last_rating.columns = ['last_rating']

    #create a new dataset for dates
    rating_count_df = pd.DataFrame({'avg_rating': avg_rating, 
                                    'num_rating': num_rating})
    rating_count_df = rating_count_df.join(last_rating)

    #turn them only one dataset
    movie_rec = movie.set_index('movie_id').join(rating_count_df)

    #rank movies by the average rating, and then by the rating counting
    ranked_movie = movie_rec.sort_values(['avg_rating', 'num_rating', 'last_rating'], 
                                         ascending=False)

    #for the border of our list, get at least movies with more than 4 reviews
    ranked_movie = ranked_movie[ranked_movie['num_rating'] >= 5]
    
    end = time()
    
    if verbose:
        print('elapsed time: {:.4f}s'.format(end-begin))
        
    return ranked_movie

#########1#########2#########3#########4#########5#########6#########7#########8
def fn_create_user_movie_dict(df_user_movie,
                              lower_filter=None,
                              verbose=False):
    '''This function creates a dictionnary structure based on an array of movies
    watched by a user.
    
    Source: Udacity Data Science Course - Lesson 6  - Identifying Reccomendations
    Forth Notebook - Class 17 - Collaborative Filtering -Recommendations with 
    MovieTweetings - Collaborative Filtering    
    
    Input:
      - df_user_movie (mandatory) - Pandas dataset with all movies vs users at
        the type user by movie, to be processed
        or a dictionnary to be filtered (if it is a dict, please informa the
        lower_filter parameter)
      - lower_filter (mandatory) - elliminate users with a number of watched
        videos below the number (Integer, default=None)
      - verbose (optional) - if you want some verbosity in your function -
        (Boolean, default=False)
    Output: 
      - movies_seen - a dictionary where each key is a user_id and the value is 
      an array of movie_ids. Creates the movies_seen dictionary
    '''
    if verbose:
        print('###function create user movie started')
        
    begin = time()
    movie_registered = dict()
    
    if type(df_user_movie) == dict: #dictionnary already processed
        if lower_filter == None or lower_filter < 1:
            if verbose:
                print('run as post processing without a lower value don´t make sense')
            return False
        else:
            if verbose:
                print('running as post processing filter, for lower:', lower_filter)
            for user, movie in df_user_movie.items():
                if len(movie) > lower_filter:
                    movie_registered[user] = movie
    else:
        if verbose:
            print('running as original dictionnary builder, for lower:', lower_filter)
        n_users = df_user_movie.shape[0]
        #creating a list of movies for each user
        for num_user in range(1, n_users+1): #added lower_filter param
            content = fn_movie_watched(df_user_movie=df_user_movie,
                                       user_id=num_user,
                                       lower_filter=lower_filter,
                                       verbose=verbose)
            if content is not None:
                movie_registered[num_user] = content #if there is some movie
            
    end = time()
    
    if verbose:
        print('elapsed time: {:.1f}s'.format(end-begin))
    
    return movie_registered

#########1#########2#########3#########4#########5#########6#########7#########8
def fn_create_user_movie(df_user_item, 
                         verbose=False):
    '''This function creates user by movie matrix. As it is normally a big data,
    please make a pre-filter on rewiew dataset, using:
    user_item = review[['user_id', 'movie_id', 'rating']]
    
    Source: Udacity Data Science Course - Lesson 6  - Identifying Reccomendations
    Forth Notebook - Class 17 - Collaborative Filtering - Recommendations with 
    MovieTweetings - Collaborative Filtering  
    
    Inputs:
      - df_user_item (mandatory) - a Pandas dataset containing all reviews for 
        movies made by users, in a condensed way (a thin dataset)
      - verbose (optional) - if you want some verbosity in your function -
        (Boolean, default=False)
    Output:
      - user_by_movie -  a thick dataset, ordered by best rated first, each row
        representing an user and each column a movie
        (a sparse dataset, as most of the movies were never rated by each user!)
    '''
    if verbose:
        print('###function create user by movie matrix started')
        
    begin = time()
    
    #remember that...
    #df_user_item = df_review[['user_id', 'movie_id', 'rating']]
    
    user_by_movie = df_user_item.groupby(['user_id', 'movie_id'])['rating'].max()  
    user_by_movie = user_by_movie.unstack()
    
    end = time()
    
    if verbose:
        print('elapsed time: {:.1f}s'.format(end-begin))
    
    return user_by_movie

#########1#########2#########3#########4#########5#########6#########7#########8
def fn_find_closest_neighbor(filt_user1,
                             limit=None,
                             verbose=True):
    '''This function takes a distance dataset and returns the closest users.
    
    Source: Udacity Data Science Course - Lesson 6  - Identifying Reccomendations
    Forth Notebook - Class 17 - Collaborative Filtering -Recommendations with 
    MovieTweetings - Collaborative Filtering

    Inputs:
      - filt_user1 (mandatory) - (Pandas dataset) the user_id of the individual 
        you want to find the closest users.
        e.g: filt_user1=df_dists[df_dists['user1'] == user]
      - limit (optional) - maximum number of closest users you want to return -
        (integer, default=None)
      - verbose (optional) - if you want some verbosity in your function -
        (Boolean, default=False)
    Output:
      - closest_neighbor - a numpy vector containing all closest users, from
        the cloest ones until the end
    '''    
    try:
        user1 = filt_user1['user1'].iloc[0]
    except TypeError:
        if verbose:
            print('you must inform a filtered dataset, see documentation')
        return False
    except IndexError:
        if verbose:
            print('an empty dataset was informed, this user does not exist')
        return False

    if verbose:
        print('###function find closest neighbors started for user', user1)
        
    begin = time()
    filt_user2 = filt_user1[filt_user1['user2'] != user1]
    closest_user = filt_user2.sort_values(by='eucl_dist')['user2']
    closest_neighbor = np.array(closest_user)
    
    if limit is not None:
        closest_neighbor = closest_neighbor[:limit]
        #if verbose:
        #    print('*limit:', len(closest_neighbor))

    end = time()
    
    if verbose:
        print('returned {} neighbors'.format(len(closest_neighbor)))
        print('elapsed time: {:.4f}s'.format(end-begin))
    
    return closest_neighbor

#########1#########2#########3#########4#########5#########6#########7#########8
def fn_find_similar_movie(df_dot_product,
                          df_movie,
                          movie_id,
                          verbose=False):
    '''This function takes one movie_id, find the physical position of it in
    our already processed dot product movie matrix and find the best similarity
    rate. Then search on the dot product matrix movies with the same rate, for
    their idxs (they are the most collinear from your movie-vector!). Finally,
    it retrieves the movie names, based on these idxs.
    
    It is similar to fn_find_closest_neighbor(), for Collaborative Filter.
    
    Source: Udacity Data Science Course - Lesson 6 - Ways to Reccomend
    Fifth Notebook - Class 21 - Content Based Recommendations
        
    Inputs:
      - df_dot_produt (mandatory) - your dot product matrix, with movies 
        collinearity values. You need to preprocess this earlier
      - df_movie (mandatory) - your movies dataset, including movie names
      - movie_id (mandatory) - a movie id to be asked for
      - verbose (optional) - if you want some verbosity in your function -
        (Boolean, default=False)
    Output:
      - similar_movies - an array of the titles of the most similar movies
    '''
    if verbose:
        print('###function find similar movies started')
        
    begin = time()
    
    if verbose:
        our_movie = df_movie[df_movie['movie_id'] == movie_id]['movie']
        print('our movie iloc: ', our_movie) #.values[0])

    #retrieve the physical position (iloc) of your movie
    movie_idx = np.where(df_movie['movie_id'] == movie_id)[0][0]
    
    #get the maximum similarity rate
    max_rate = np.max(df_dot_product[movie_idx])
    if verbose:
        print('*max collinearity:', max_rate)
    
    #retrieve the movie indices for maximum similarity
    similar_idx = np.where(df_dot_product[movie_idx] == max_rate)[0]
    
    #make an array, locating by position on dataset, the similars + title
    similar_movie = fn_get_movie_name(df_movie=df_movie,
                                      movie_id=similar_idx,
                                      by_id=False,
                                      as_list=False,
                                      verbose=verbose)
    end = time()
    
    if verbose:
        print('similar movies:', len(similar_movie))
        print('elapsed time: {:.4f}s'.format(end-begin))
    
    return similar_movie

#########1#########2#########3#########4#########5#########6#########7#########8
def fn_get_movie_name(df_movie,
                      movie_id,
                      by_id=True,
                      as_list=True,
                      verbose=False):
    '''This function finds a movie (or a list of movies) by its index and return
    their titles as a list.
    
    There are only two ways implemented at this moment:
    
    First, it takes a single movie id and returns a single name, inside a list
    (function default)
    
    Second, it takes a list of movies idx and returns a numpy Array, with
    multiple names.
    (for providing service for fn_find_similar_movie())
    
    Source: Udacity Data Science Course - Lesson 6 - Ways to Reccomend
    Fifth Notebook - Class 21 - Content Based Recommendations
    
    Inputs:
      - df_movie (mandatory) - a preformated dataset of movies
      - movie_id (mandatory) - a list of movie_id or one single movie_id 
        (depends on the case)
      - verbose (optional) - if you want some verbosity in your function -
        (Boolean, default=False)
    Output:
      - movie - a list of names of movies
    '''
    if verbose:
        print('###function get movie names started')
        
    begin = time()
    
    if as_list:
        if by_id:
            if verbose:
                print('taking movies id, returning a list')
                print('*function default')
            try:
                movie_list = list(df_movie[df_movie['movie_id'].isin(movie_id)]['movie'])
            except TypeError:
                movie_id = [movie_id]
                movie_list = list(df_movie[df_movie['movie_id'].isin(movie_id)]['movie'])
        else:
            if verbose:
                print('taking a movies idx, returning a list') #id=idx
                print('not yet implemented!')
            return False
    else:
        if by_id:
            if verbose:
                print('taking a movies id, returning numpy vector')
                print('not yet implemented!')
            return False
        else:
            if verbose:
                print('taking movies idx, returning a list') 
                print('*default for fn_find_similar_movie') #id=idx
            movie_list = np.array(df_movie.iloc[movie_id]['movie']) 
            
    end = time()
    
    if verbose:
        print('elapsed time: {:.6f}s'.format(end-begin))
   
    return movie_list

#########1#########2#########3#########4#########5#########6#########7#########8
def fn_make_recommendation_collab(filt_dist,
                                  df_user_movie,
                                  df_movie,
                                  num_rec=10,
                                  limit=None,
                                  min_rating=7,
                                  sort=False,
                                  verbose=False):
    
    '''This function takes filtered distances from a focus user and find the
    closest users. Then retrieves already watched movies by the user, just to
    not recommend them. And in sequence, iterate over closest neighbors, retrieve
    their liked movies and it they were not already seen, put in a recommend
    list. For finish, transform movies ids into movies names and return a list.
    
    Source: Udacity Data Science Course - Lesson 6  - Identifying Reccomendations
    Forth Notebook - Class 17 - Collaborative Filtering - Recommendations with 
    MovieTweetings - Collaborative Filtering
    
    Input:
      - filt_dist (mandatory) - a user by movie type dataset, created with the 
        function fn_create_user_item(), from user distances dataset - 
        (Pandas dataset)
        e.g.: filt_dist=df_dist[df_dist['user1'] == user]
      - df_user_movie (mandatory) - dataset in the shape user by movie -
        (Pandas dataset)
      - df_movie (mandatory) - dataset in she shape movies - 
        (Pandas dataset)
      - num_rec (optional) - (int) number of recommended movies to return
      - limit (optional) - extra parameter for fn_find_closest_neighbor - 
        it limits the number of neighbors (normally 100 is more than enough) 
      - min_rating (optional) - extra parameter for fn_movie_liked2() - it is
        the worst score for considering a movie as liked (normally rate 7 is 
        enough)
      - sort (optional) - extra parameter for fn_movie_liked2() - if you want
        to show the best rated movies first (in this algorithm it is useless)
      - verbose (optional) - if you want some verbosity in your function -
        (Boolean, default=False)
    Output:
      - recommendations - a list of movies - if there are "num_recs" recommendations return this many
                          otherwise return the total number of recommendations available for the "user"
                          which may just be an empty list
    '''
    try:
        user_id = filt_dist['user1'].iloc[0]
    except TypeError:
        if verbose:
            print('you must inform a filtered dataset, see documentation')
        return False
    except IndexError:
        if verbose:
            print('an empty dataset was informed, this user does not exist')
        return False

    if verbose:
        print('###function make recommendations started')
        
    begin = time()
    
    # movies_seen by user (we don't want to recommend these)
    movie_user = df_user_movie.loc[user_id].dropna()    
    movie_seen = movie_user.index.tolist() 
    
    if verbose:
        print('*seen {} movies by user {} - first is {}'.format(len(movie_seen), user_id, movie_seen[0]))
    
    closest_neighbor = udacourse3.fn_find_closest_neighbor(filt_user1=filt_dist,
                                                           limit=limit,
                                                           verbose=verbose)
    if verbose:
        print('*{} closest neigbors'.format(len(closest_neighbor)))
        
    #creating your recommended array
    rec = np.array([])
    
    #from closest neighbors, 1-take move & 2-that had not been watched
    for i in range (0, len(closest_neighbor)):
        neighbor_id = closest_neighbor[i]
        filt_user = df_user_movie.loc[neighbor_id].dropna()
        if verbose:
            print('comparing with neighbor', neighbor_id)
        neighbor_like = udacourse3.fn_movie_liked2(item=filt_user,
                                                   min_rating=7,
                                                   sort=False,
                                                   verbose=False)
        if verbose:
            print('...that liked {} movies'.format(len(neighbor_like)))            

        #take recommendations by difference
        new_rec = np.setdiff1d(neighbor_like, 
                               movie_seen, 
                               assume_unique=True)
        if verbose:
            print('...and so, {} were new!'.format(len(new_rec)))
                
        #store rec
        rec = np.unique(np.concatenate([new_rec, rec], axis=0))
        if verbose:
            print('...now I have {} movies in my record'.format(len(rec)))
            #print(rec)
            #print()
         
        #if store is OK, exit the engine
        if len(rec) > num_rec-1:
            break
    
    #take the titles
    recommendation = udacourse3.fn_movie_name(df_movie=df_movie,
                                              movie_id=rec,
                                              verbose=False)
    end = time()
    
    if verbose:
        print('elapsed time: {:.4f}s'.format(end-begin))
    
    return recommendation

#########1#########2#########3#########4#########5#########6#########7#########8
#########1#########2#########3#########4#########5#########6#########7#########8
def fn_make_recommendation_content(df_dot_product,
                                   df_movie,
                                   user,
                                   verbose=False):
    '''This function...
    
    Source: Udacity Data Science Course - Lesson 6 - Ways to Reccomend
    Fifth Notebook - Class 21 - Content Based Recommendations
    
    Input:
      - user (mandatory) - 
      - verbose (optional) - if you want some verbosity in your function -
        (Boolean, default=False)
    Output:
      - recommendation - a Python dictionary containing user as keys and
        values for recommendations
    '''
    if verbose:
        print('###function make recommendations started')
        
    begin = time()

    # Create dictionary to return with users and ratings
    recommendation = defaultdict(set)
    # How many users for progress bar
    try:
        num_user = len(user)
    except TypeError:
        user = [user]
        if verbose:
            print('only one user was informed, putting it in a list')
        num_user = len(user)

    # Create the progressbar
    counter = 0
    bar = progressbar.ProgressBar(maxval=num_user+1, 
                                  widgets=[progressbar.Bar('=', '[', ']'), 
                                                           ' ',
                                                           progressbar.Percentage()])
    bar.start()
    
    #iterate user by user
    for one_user in user:
        
        #only updating the progress bar
        counter += 1 
        bar.update(counter)

        #taking only the reviews seen by this user
        review_temp = ranked_review[ranked_review['user_id'] == one_user]
        movie_temp = np.array(review_temp['movie_id'])
        movie_name = np.array(udacourse3.fn_get_movie_name(
                                  df_movie=movie,
                                  movie_id=movie_temp,
                                  verbose=verbose))

        #iterate each of these movies (highest ranked first) 
        #taking only the movies that were not watched by this user
        #and that are most similar - these are elected to be the
        #recommendations - I need only 10 of them!
        #you keep going until there are no more movies in the list
        #of this user
        for movie_id in movie_temp:
            recommended_movie = udacourse3.fn_find_similar_movie(
                                    df_dot_product=dot_prod_movie,
                                    df_movie=df_movie,
                                    movie_id=movie_id,
                                    verbose=True)
            
            temp_recommendation = np.setdiff1d(recommended_movie, movie_name)
            
            recommendation[one_user].update(temp_recommendation)

            #if you have the enough number of recommendations, you will stop
            if len(recommendation[one_user]) >= 10:
                break

    bar.finish()
    end = time()
    
    if verbose:
        print('elapsed time: {:.2f}s'.format(end-begin))
    
    return recommendation

#########1#########2#########3#########4#########5#########6#########7#########8
def fn_movie_liked2(item, 
                    min_rating=7,
                    sort=True,
                    verbose=False):
    '''This function takes all the items for one user and return the best rated
    items.
    
    Source: Udacity Data Science Course - Lesson 6  - Identifying Reccomendations
    Forth Notebook - Class 17 - Collaborative Filtering -Recommendations with 
    MovieTweetings - Collaborative Filtering

    Inputs:
      - item (mandatory) - a dataset on the shape user by movie, filtered 
        for an individual user.
        e.g. item=user_by_movie.loc[user_id].dropna()  
      - min_rating (optional) - the trigger point to consider an item to be 
        considered "nice" for an user (integer, default=7)
      - sort (optional) - if you want to show the most rated items first
        (Boolean, default=True)
      - verbose (optional) - if you want some verbosity in your function -
        (Boolean, default=False)
    Output:
      - movie_liked - an array of movies the user has watched and liked
    '''
    user = item.name

    if verbose:
        print('###function movies liked started for user', user)

    begin = time()
    movie_liked = item[item > 7]
    
    if sort:
        if verbose:
            print('*sorting the list - best rated first')
        movie_liked = movie_liked.sort_values(ascending=False)

    movie_liked = np.array(movie_liked.index)
    
    end = time()
    
    if verbose:
        print('elapsed time: {:.4f}s'.format(end-begin))
    
    return movie_liked

#########1#########2#########3#########4#########5#########6#########7#########8
def fn_movie_name(df_movie,
                  movie_id,
                  verbose=False):
    '''This function takes a list of movies_id (liked from other user) and
    returns ne movie titles
    
    Source: Udacity Data Science Course - Lesson 6  - Identifying Reccomendations
    Forth Notebook - Class 17 - Collaborative Filtering -Recommendations with 
    MovieTweetings - Collaborative Filtering    
    
    Inputs:
      - df_movie (mandatory) - the movies dataset - Pandas Dataset
      - movie_id (mandatory) - a numpy vector containing a list of movie_ids
      - verbose (optional) - if you want some verbosity in your function -
        (Boolean, default=False)
    Output:
      - movies - a list of movie names associated with the movie_ids (python
        list)
    '''
    if verbose:
        print('###function movie names started')
        
    begin = time()
    
    movie_get = df_movie[df_movie['movie_id'].isin(movie_id)]
    movie_list = movie_get['movie'].to_list()
    
    end = time()
    
    if verbose:
        print('elapsed time: {:.4f}s'.format(end-begin))
   
    return movie_list

#########1#########2#########3#########4#########5#########6#########7#########8
def fn_movie_watched(df_user_movie,
                     user_id,
                     lower_filter=None,
                     verbose=False):
    '''This function creates a array structure. Keys are users, content is
    a list of movies seen. DF is a user by movie dataset, created with the
    function fn_create_user_item.
    
    Source: Udacity Data Science Course - Lesson 6  - Identifying Reccomendations
    Forth Notebook - Class 17 - Collaborative Filtering -Recommendations with 
    MovieTweetings - Collaborative Filtering

    Input:
      - df_user_movie (mandatory) df_user_movie (mandatory) - a user by movie 
        type dataset, created with the function fn_create_user_item(), from 
        user (Pandas dataset)
      - user_id (mandatory) - the user_id of an individual as int
      - lower_filter (optional) - elliminate users with a number of watched
        videos below the number (Integer, default=None)      
      - verbose (optional) - if you want some verbosity in your function -
        (Boolean, default=False)
    Output:
      - movies - an array of movies the user has watched
    '''
    if verbose:
        print('###function movies watched started')
        
    begin = time()
    
    movie = df_user_movie.loc[user_id][df_user_movie.loc[user_id].isnull() == False].index.values
    
    end = time()
    if verbose:
        print('elapsed time: {:.4f}s'.format(end-begin))
        
    if lower_filter == None:
        return movie
    elif lower_filter > 0:
        if len(movie) > lower_filter:
            return movie
        else:
            return None
    else:
        raise Exception('something went wrong with lower filter parameter')

#########1#########2#########3#########4#########5#########6#########7#########8
def fn_popular_recommendation(user_id, 
                              num_top, 
                              ranked_movie,
                              verbose=False):
    '''this function makes a list of top recommended movies, by title. Laterly 
    this function can be forked for other purposes.
    
    Source: Udacity Data Science Course - Lesson 6 - Recomendation Engines - 
    Class 8 - Second Notebook - Intro to Recommendation data - Finding Most 
    Popular Movies.

    Input:
      - user_id (mandatory) - the user_id (str) of the individual you are making 
        recommendations for
      - num_top (mandatory) - an integer of the number recommendations you want
        back
      - ranke_movie (mandatory) - a pandas dataframe of the already ranked 
        movies based on avg rating, count, and time
      - verbose (optional) - if you want some verbosity in your function -
        (Boolean, default=False)
    Output:
      - top_movies - a Pandas Series of the num_top recommended movies by movie 
        title in order, from the best to the worst.
    '''
    if verbose:
        print('###function popular recomendations started')
        
    begin = time()

    top_movie = list(ranked_movie['movie'][:num_top])

    end = time()
    
    if verbose:
        print('elapsed time: {:.6f}s'.format(end-begin))

    return top_movie

#########1#########2#########3#########4#########5#########6#########7#########8
def fn_popular_recommendation_filtered(user_id, 
                                       num_top, 
                                       ranked_movie, 
                                       year=None, 
                                       genre=None,
                                       verbose=False):
    '''This function creates some filter for adding robustness for our model.
    Laterly this function can be forked for other purposes.
    
    Source: Udacity Data Science Course - Lesson 6 - Recommendation Engines - 
    Class 5 - First Notebook - Intro to Recommendation data - Part II - Adding
    Filters.
    
    Inputs:
      - user_id (mandatory) - the user_id (str) of the individual you are making 
        recommendations for
      - num_top (mandatory) - an integer of the number recommendations you want
        back
      - ranked_movie (mandatory) - a pandas dataframe of the already ranked movies
        based on average rating, count, and time
      - year (mandatory) - a list of strings with years of movies
      - genre (mandatory) - a list of strings with genres of movies
      - verbose (optional) - if you want some verbosity in your function -
        (Boolean, default=False)
    Output:
      - top_movies - a list of the num_top recommended movies by movie title in 
        order from the best to worst one
    '''    
    if verbose:
        print('###function popular recommendations (filtered) started')
        
    begin = time()

    #a year filter
    if year is not None:
        if verbose:
            print('*year filter activated')
        ranked_movie = ranked_movie[ranked_movie['date'].isin(year)]

    #a genre filter    
    if genre is not None:
        if verbose:
            print('*genre filter activated')
        num_genre_match = ranked_movie[genre].sum(axis=1)
        #at least one was found!
        ranked_movie = ranked_movie.loc[num_genre_match >= 1, :] 
                  
    #recreate a top list for movies (now filtered!)
    #num_top is the cutting criteria!
    top_movie = list(ranked_movie['movie'][:num_top]) 

    end = time()
    
    if verbose:
        print('elapsed time: {:.4f}s'.format(end-begin))

    return top_movie

#########1#########2#########3#########4#########5#########6#########7#########8
def fn_take_correlation(for_user1, 
                        for_user2,
                        verbose=False):
    '''This function takes two movies series from a dataset like 
    movies_to_analyze and returns the correlation between these users.

    Important Observation:
    The main dataset is normally too large to be passed into the function. So
    you need to make a pre-filter and feed this function ONLY with the data
    of the two users in focus. 
    Example of filter: for_user1=user_by_movie.loc[2] <- 2 is the ID of user1
        
    Source: Udacity Data Science Course - Lesson 6  - Identifying Reccomendations
    Forth Notebook - Class 17 - Collaborative Filtering -Recommendations with 
    MovieTweetings - Collaborative Filtering    
    
    Input:
      - for_user1 (mandatory) - raw series of movies data for User 1. It can 
        contain NaN (Pandas Series) 
      - for_user2 (mandatory) - raw series of movies data for User 2. It can 
        contain NaN (Pandas Series) 
      - verbose (optional) - if you want some verbosity in your function -
        (Boolean, default=False)
    Output:
      - the correlation between the matching ratings between the two users
    '''
    if verbose:
        print('###function take correlation coefficient started')
        
    begin = time()

    #find the movie list for each user
    usr1_movie = for_user1.dropna().index
    usr2_movie = for_user2.dropna().index

    #getting the insersections
    sim_mov = np.intersect1d(usr1_movie, usr2_movie, assume_unique=True)
    
    #finding the weights for the insersection movies
    #sim_mov=[1454468, 1798709, 2883512]] for user1=2 and user2=66
    sr1 = for_user1.loc[sim_mov] 
    sr2 = for_user2.loc[sim_mov]
    
    correlation = fn_compute_correlation(x=sr1, 
                                         y=sr2,
                                         corr_type='intersection',
                                         verbose=verbose)    
    end = time()
    
    if verbose:
        print('elapsed time: {:.4f}s'.format(end-begin))
    
    return correlation

#########1#########2#########3#########4#########5#########6#########7#########8
def fn_take_euclidean_dist(for_user1, 
                           for_user2,
                           verbose=False):
    '''This function takes two movies series from a dataset like 
    movies_to_analyze and returns the correlation between these users.

    Important Observation:
    The main dataset is normally too large to be passed into the function. So
    you need to make a pre-filter and feed this function ONLY with the data
    of the two users in focus. 
    Example of filter: for_user1=user_by_movie.loc[2] <- 2 is the ID of user1
        
    Source: Udacity Data Science Course - Lesson 6  - Identifying Reccomendations
    Forth Notebook - Class 17 - Collaborative Filtering -Recommendations with 
    MovieTweetings - Collaborative Filtering    
    
    Input:
      - for_user1 (mandatory) - raw series of movies data for User 1. It can 
        contain NaN (Pandas Series) 
      - for_user2 (mandatory) - raw series of movies data for User 2. It can 
        contain NaN (Pandas Series) 
      - verbose (optional) - if you want some verbosity in your function -
        (Boolean, default=False)
    Output:
      - the euclidean distance between the two users
    '''
    if verbose:
        print('###function take euclidean distance started')
        
    begin = time()
    
    #find the movie list for each user
    try:
        usr1_movie = for_user1.dropna().index
    except AttributeError:
        if verbose:
            print('you need to give a dataset row, see the function documentation')
        return False
    try:
        usr2_movie = for_user2.dropna().index
    except AttributeError:
        if verbose:
            print('you need to give a dataset row, as for_user2=user_by_movie.loc[66]')
        return False
    
    #getting the insersections
    sim_mov = np.intersect1d(usr1_movie, usr2_movie, assume_unique=True)
    
    #finding the weights for the insersection movies
    #sim_mov=[1454468, 1798709, 2883512]] for user1=2 and user2=66
    sr1 = for_user1.loc[sim_mov] 
    sr2 = for_user2.loc[sim_mov]
        
    euclidean_distance = fn_calculate_distance(x=sr1, 
                                               y=sr2,
                                               dist_type='euclidean',
                                               verbose=verbose) 
    end = time()
    
    if verbose:
        print('elapsed time: {}s'.format(end-begin))
    
    return euclidean_distance

###Stats functions##############################################################
#########1#########2#########3#########4#########5#########6#########7#########8
def fn_experiment_size(p_null, 
                       p_alt, 
                       alpha=0.05, 
                       beta=0.20,
                       verbose=False):
    '''This function takes a size of effect and returns the minimum number of 
    samples needed to achieve the desired power.
    
    Source: Udacity Data Science Course - Lesson 4 - Statistical Considerations
    into testing - Second Notebook - Experiment Size - Analytic Solution.
    
    Inputs:
      - p_null (mandatory) - null hypothesis success rate (base) - (numpy Float)
      - p_alt (mandatory) - success rate (desired) - what we want to detect -
        (numpy Float)
      - alpha (optional) - Type-I (false positive) rate of error - (numpy Float -
        default=5%)
      - beta (optional) - Type-II (false negative) rate of error - (numpy Fload -
        default=20%)
      - verbose (optional) - if you want some verbosity in your function -
        (Boolean, default=False)
    Output:
      - n - required number of samples for each group, in order to obtain the 
        desired power (it is considered that the share for each group is 
        equivalent)
    '''
    if verbose:
        print('###function experiment size started - Analytic solution')
        
    begin = time()

    #takes z-scores and st dev -> 1 observation per group!
    z_null = stats.norm.ppf(1-alpha)
    z_alt  = stats.norm.ppf(beta)
    sd_null = np.sqrt(p_null * (1-p_null) + p_null * (1-p_null))
    sd_alt  = np.sqrt(p_null * (1-p_null) + p_alt  * (1-p_alt) )
    
    #calculate the minimum sample size
    p_diff = p_alt - p_null
    num = ((z_null*sd_null - z_alt*sd_alt) / p_diff) ** 2
    
    num_max = np.ceil(num)
    
    end = time()
    
    if verbose:
        print('elapsed time: {:.4f}s'.format(end-begin))
        print('experiment size:', num_max)

    return num_max

#########1#########2#########3#########4#########5#########6#########7#########8
def fn_invariant_analytic(df,
                          p=0.5,
                          value=0.0,
                          verbose=False):
    '''This function checks a invariant metric by analytic approach. One example
    of a invariant is if the division between two different webpages (one for H0
    and the other for H1) is similar.
    
    Source: Udacity Data Science Course - Lesson 4 - Statistical Considerations 
    into testing - Class 3 - First Notebook - Statistical Significance.
    
    Inputs:
      - df (mandatory) - dataset containing the binary data to be checked
      - p (optional) - probability (default=50%)
      - value (optional) - central value (default=0)
      - verbose (optional) - if you want some verbosity in your function -
        (Boolean, default=False)
    Output:
      - It is a report function only! Returns True if everything runned well
    '''
    if verbose:
        print ('###function for invariant check started - analytic approach')
        
    begin = time()
    
    #number of trials
    num_observation = df.shape[0]
    if verbose:
        print('number of observations:', num_observation)
    
    #number of successes
    num_control = df[df['condition'] == 0].shape[0] #data.groupby('condition').size()[0]
    if verbose:
        print('number of cases with success:', num_control)
        
    #z-score and p-value
    st_dev = np.sqrt(num_observation * p * (1-p)) #formula for binomial distribution
    if verbose:
        print('Standard Deviation for Binomial Distribution: {:.1f}'.format(st_dev))
    
    z_score = ((num_control + 0.5) - p * num_observation) / st_dev
    if verbose:
        print('z-score: {:.4f}'.format(z_score))
        
    #cumulative distribution function
    cdf = stats.norm.cdf(0)
    if verbose:
        print('cumulative values over {} is {:.4f} ({:.2f}%)'\
              .format(value, cdf, cdf*100))
    
    #analytic p-value
    p_value = 2 * stats.norm.cdf(z_score)
    
    end = time()
    if verbose:
        print()
        print('analytic p-value: {:.4f} ({:.2f}%)'.format(p_value, p_value*100))
        print('*a low value means that H0 don´t have a good explanatory power')
        print('elapsed time: {:.4f}s'.format(end-begin))
        
    return True
        
#########1#########2#########3#########4#########5#########6#########7#########8
def fn_invariant_simulated(df,
                           p=0.5,
                           num_trials=200_000,
                           verbose=False):
    '''This function checks a invariant metric by a simulation approach.
    
    Source: Udacity Data Science Course - Lesson 4 - Statistical Considerations 
    into testing - Class 3 - First Notebook - Statistical Significance.
    
    Inputs:
      - df (mandatory) - dataset containing the binary data to be checked
      - p (optional) - probability (default=50%)
      - num_trials - number of trials for randomly simulating the experiment
      - verbose (optional) - if you want some verbosity in your function -
        (Boolean, default=False)
    Output:
      - It is a report function only! Returns True if everything runned well
    '''
    if verbose:
        print ('###function for invariant check started - simulation approach')
        
    begin = time()
    
    #number of trials
    num_observation = df.shape[0]
    if verbose:
        print('number of observations:', num_observation)        
        
    #number of successes
    num_control = df[df['condition'] == 0].shape[0] #data.groupby('condition').size()[0]
    if verbose:
        print('number of cases with success:', num_control)

    #simulate outcomes under null, compare to observed outcome
    samples = np.random.binomial(num_observation, p, num_trials)
        
    if verbose:
        print()
        print('*simulation part')
        print('simulated samples: {}'.format(len(samples)))
            
    #number of samples below control number
    samples_below = sum(samples <= num_control)
    if verbose:
        print('number of cases below control:', samples_below) 
    
    #samples above control number
    samples_above = sum(samples >= (num_observation-num_control))
    
    #simulated p-value
    p_value = np.logical_or(samples <= num_control, samples >= (num_observation-num_control)).mean()
    
    end = time()
    
    if verbose:
        print()
        print('simulated p_value: {:.4f} ({:.2f}%)'\
              .format(p_value, p_value*100))
        print('*a low value means that H0 don´t have a good explanatory power')
        print('elapsed time: {:.4f}s'.format(end-begin))
        
    return True
        
#########1#########2#########3#########4#########5#########6#########7#########8
def fn_peeking_correction(alpha=0.05, 
                          p_success=0.5, 
                          num_trials=1000, 
                          num_blocks=2, 
                          num_sims=10000,
                          verbose=False):
    '''This function make a estimative of the individual error rate necessary
    to limit the Type I error (false positive) rate, if an early stopping decision
    is made. It uses a simulation, to predict if significant result could exist
    when peeking ahead.
    
    Source: Udacity Data Science Course - Lesson 4 - Statistical Considerations
    into testing - Class 14 - Fifth Notebook - Early Stopping - with Multiple 
    Corrections.
    
    Inputs:
      - alpha (optional) - overall Type I error (false positive) rate that was 
        desired for the experiment - (numpy Float - default=5%) 
      - p_success (optional): probability of obtaining a success on an individual
        trial - (numpy Float - default=-50%)
      - num_trials (optional): number of trials that runs in a full experiment
        (Integer - default=10000)
      - num_blocks (optional): number of times that a a data is looked for
        (including the end) - (Integer - default=2)
      - num_sims (optional) - number of times that the simulated experiments run
        (Integer - default=10000)
    Output:
        alpha_individual: Individual error rate required to achieve overall error 
        rate
    '''
    if verbose:
        print('###function peeking correction started')

    begin=time()

    #data generation
    trials_per_block = np.ceil(num_trials / num_blocks).astype(int)
    try:
        data = np.random.binomial(trials_per_block, 
                                  p_success, 
                                  [num_sims, num_blocks])
    except ValueError:
        print('something went wrong doing binomials - p seems to be invalid!')
        return (trials_per_block, p_success, [num_sims, num_blocks])
    
    #data standardization
    data_cumsum = np.cumsum(data, axis = 1)
    block_sizes = trials_per_block * np.arange(1, num_blocks+1, 1)
    block_means = block_sizes * p_success
    block_sds   = np.sqrt(block_sizes * p_success * (1-p_success))
    data_zscores = (data_cumsum - block_means) / block_sds
    
    #the necessary individual error rate
    max_zscores = np.abs(data_zscores).max(axis = 1)
    z_crit_ind = np.percentile(max_zscores, 100 * (1 - alpha))
    alpha_individual = 2 * (1 - stats.norm.cdf(z_crit_ind))
 
    end = time()
    
    if verbose:
        print('probabilities - alpha individual: {:.4f} ({:.2f}%)'\
              .format(alpha_individual, alpha_individual*100))
        print('elapsed time: {:.4f}s'.format(end-begin))

    return alpha_individual

#########1#########2#########3#########4#########5#########6#########7#########8
def fn_peeking_simulation(alpha=0.05, 
                          p=0.5, 
                          num_trials=1000,
                          num_blocks=2,
                          num_sims=10_000,
                          verbose=False):
    '''This function aims to simulate the rate of Type I error (false positive) 
    produced by the early stopping decision. It is based on a significant result
    when peeking ahead.
    
    Source: Udacity Data Science Course - Lesson 4 - Statistical Considerations
    into testing - Class 15 - Fifth Notebook - Early Stopping.
    
    Inputs:
        - alpha (optional) - Type I error rate that was supposed
        - p (optional) - probability of individual trial success
        - num_trials (optional) - number of trials in a full experiment
        - num_blocks (optional) - number of times data is looked at (including end)
        - num_sims: Number of simulated experiments run
    Output:
        p_sig_any: proportion of simulations significant at any check point, 
        p_sig_each: proportion of simulations significant at each check point
    '''
    if verbose:
        print('###function peeking for early stopping started - Simulating {} times'\
              .format(num_sims))
        
    begin=time()
    
    #generate the data
    trials_per_block = np.ceil(num_trials / num_blocks).astype(int)
    data = np.random.binomial(trials_per_block, p, [num_sims, num_blocks])
    
    #put the data under a standard
    data_cumsum = np.cumsum(data, axis=1) #cumsum is s summation 
    block_sizes = trials_per_block * np.arange(1, num_blocks+1, 1)
    block_means = block_sizes * p
    block_sds = np.sqrt(block_sizes * p * (1-p))
    data_zscores = (data_cumsum - block_means) / block_sds
    
    #results
    z_crit = stats.norm.ppf(1-alpha/2) #norm is a normal distribution
    sig_flags = np.abs(data_zscores) > z_crit
    p_sig_any = (sig_flags.sum(axis=1) > 0).mean()
    p_sig_each = sig_flags.mean(axis=0)
    
    tuple = (p_sig_any, p_sig_each)
    
    end = time()
    
    if verbose:
        print('probabilities - signal(any): {:.4f} ({:.2f}%), signal(each): {}'\
              .format(p_sig_any, p_sig_any*100, p_sig_each))
        print('elapsed time: {:.4f}s'.format(end-begin))
    
    return tuple

#########1#########2#########3#########4#########5#########6#########7#########8
def fn_power(p_null, 
             p_alt, 
             num, 
             alpha=0.05, 
             plot=False,
             verbose=False):
    '''This function takes an alpha rate and computes the power of detecting the 
    difference in two populations.The populations can have different proportion 
    parameters.
    
    Source: Udacity Data Science Course - Lesson 4 - Statistical Considerations
    into testing - Class 5 - Second Notebook - Experiment Size - By Trial and 
    Error.
    
    Inputs:
      - p_null (mandatory) - rate of success (base) under the Null hypothesis
        (numpy Float) 
      - p_alt (mandatory) -  rate of sucess (desired) must be larger than the
        first parameter - (numpy Float)
      - num (mandatory) - number of observations for each group - (integer)
        alpha (optional) - rate of Type-I error (false positive-normally the
        more dangerous) - (numpy Float - default 5%)
      - plot (optional) - if you want to plot the distribution - (Boolean, 
        default=False)
      - verbose (optional) - if you want some verbosity in your function -
        (Boolean, default=False)
    Output:
        power - the power for detection of the desired difference under the 
        Null Hypothesis.
    '''
    if verbose:
        print('###function power of an Hypothesis started - by Trial  & Error')
        
    begin = time()
    
    #the idea: start with the null hypothesis. Our main target is to find 
    #Type I errors (false positive) trigger (critical value is given by
    #Alpha parameter - normally 5%).
    
    #se_null → standard deviation for the difference in proportions under the
    #null hypothesis for both groups
    #-the base probability is given by p_null
    #-the variance of the difference distribution is the sum of the variances for
    #-the individual distributions
    #-for each group is assigned n observations.
    se_null = np.sqrt((p_null * (1-p_null) + p_null * (1-p_null)) / num)
    #null_dist → normal continuous random variable (form Scipy doc)
    null_dist = stats.norm(loc=0, scale=se_null)

    #p_crit: Compute the critical value of the distribution that would cause us 
    #to reject the null hypothesis. One of the methods of the null_dist object 
    #will help you obtain this value (passing in some function of our desired 
    #error rate alpha). The power is the proportion of the distribution under 
    #the alternative hypothesis that is past that previously-obtained critical value.
    p_crit = null_dist.ppf(1-alpha) #1-alpha=95%
    
    #se_alt: Now it's time to make computations in the other direction. 
    #This will be standard deviation of differences under the desired detectable 
    #difference. Note that the individual distributions will have different variances 
    #now: one with p_null probability of success, and the other with p_alt probability of success.
    se_alt  = np.sqrt((p_null * (1-p_null) + p_alt  * (1-p_alt)) / num)

    #alt_dist: This will be a scipy norm object like above. Be careful of the 
    #"loc" argument in this one. The way the power function is set up, it expects 
    #p_alt to be greater than p_null, for a positive difference.
    alt_dist = stats.norm(loc=p_alt-p_null, scale=se_alt)

    #beta → Type-II error (false negative) - I fail to reject the null for some
    #non-null states
    beta = alt_dist.cdf(p_crit)    
    
    if plot:
        plot = fn_plot(first_graph=null_dist, 
                       second_graph=alt_dist,
                       aux=p_crit,
                       type='htest',
                       verbose=verbose)
        
    power = (1 - beta)
    end = time()
    
    if verbose:
        print('hypotesis power coefficient: {:.4f} ({:.2f}%)'.format(power, power*100))
        print('*a low value means that H0 don´t have a good explanatory power')
        print('elapsed time: {:.4f}s'.format(end-begin))
        
    return power

#########1#########2#########3#########4#########5#########6#########7#########8
def fn_quantile_ci(data, 
                   q, 
                   c=0.95, 
                   num_trials=1000,
                   plot=False,
                   verbose=False):
    '''This function takes a quartile for a data and returns a confidence 
    interval, using Bootstrap method.
    
    Source: Udacity Data Science Course - Lesson 4 - Statistical Considerations
    into testing - Class 8 - Third Notebook - Non-Parametric Tests - Bootstrapping 
    Confidence Intervals using Quantiles.
    
    Inputs:
      - data (mandatory) - a series of numpy Float data to be processed - it
        can be a Pandas Series - (numpy Array)
      - q (mandatory) - quantile to be estimated - (numpy Array - between 0 and 1)
      - c (optional) - confidence interval - (float, default: 95%)
      - num_trials (optional) - the number of samples that bootstrap will perform
        (default=1000)
      - plot (optional) - if you want to plot the distribution - (Boolean, 
        default=False)
      - verbose (optional) - if you want some verbosity in your function -
        (Boolean, default=False)
    Output:
      - ci - upper an lower bound for the confidence interval - (Tuple of numpy Float)
    '''
    if verbose:
        print("###function quantile confidence interval started - Bootstrapping method")
        
    begin=time()
    
    if plot:
        if verbose:
            print('initial histogram distribution graph')
        plot = fn_plot(first_graph=data, 
                       type='hist',
                       verbose=verbose)

    #sample quantiles for bootstrap
    num_points = data.shape[0]
    sample_qs = []
    
    #loop for each bootstrap element
    for _ in range(num_trials):
        #random sample for the data
        sample = np.random.choice(data, num_points, replace=True) #with replacement
        
        #desired quantile
        sample_q = np.percentile(sample, 100 * q)
        
        #append to the list of sampled quantiles
        sample_qs.append(sample_q)
        
    #confidence interval bonds
    lower_limit = np.percentile(sample_qs, (1 - c) / 2 * 100)
    upper_limit = np.percentile(sample_qs, (1 + c) / 2 * 100)
    
    tuple = (lower_limit, upper_limit)
    end = time()
    
    if verbose:
        print('confidence interval - lower: {:.4f} ({:.2f}%) upper: {:.4f} ({:.2f}%)'\
              .format(lower_limit, lower_limit*100, upper_limit, upper_limit*100))
        print('elapsed time: {:.4f}s'.format(end-begin))

    return (lower_limit, upper_limit)

#########1#########2#########3#########4#########5#########6#########7#########8
def fn_quantile_permutation_test(x, 
                                 y, 
                                 q, 
                                 alternative='less',
                                 num_trials=10_000,
                                 plot=False,
                                 verbose=False):
    '''this function takes a vector of independent feature, another of dependent
    feature and calculates a confidence interval for a quantile of a dataset.
    It uses a Bootstrap method.
    
    Source: Udacity Data Science Course - Lesson 4 - Statistical Considerations
    into testing - Class 8 - Third Notebook - Non-Parametric Tests - Bootstrapping
    Confidence intervals using Permutation Test.
    
    Inputs:
      - x (mandatory) - a vector containing zeroes and 1 values, for the 
        independent (to be grouped) feature - (Boolean)
      - y (mandatory) - a vector containing zeroes and 1 values, for the 
        dependent (output) feature
      - q (mandatory) - a verctor containing zeroes and 1 valures for the output
        quantile
      - alternative (optional) - please inform the type of test to be performed
        (possible: 'less' and 'greater') - (default='less')
      - num_trials (optional) number of permutation trials to perform 
      - plot (optional) - if you want to plot the distribution - (Boolean, 
        default=False)
      - verbose (optional) - if you want some verbosity in your function -
        (Boolean, default=False)
    Output:
      - the estimated p-value of the test (numpy Float)
    '''
    if verbose:
        print("###function quantile permutation test - Bootstrapping method")
        
    begin=time()
    
    if plot:
        if verbose:
            print('first showing histogram graphs for both condition 0 and condition 1')
        df_plot = pd.concat([y, x], axis=1, join="inner") #recreate dataset
        plot = fn_plot(first_graph=df_plot[df_plot['condition'] == 0]['time'], 
                       second_graph=df_plot[df_plot['condition'] == 1]['time'],
                       aux=df_plot['time'],
                       type='2hist',
                       verbose=verbose)      
          
    #initialize list for bootstrapped sample quantiles
    sample_diffs = []
    
    #loop on trials
    for _ in range(num_trials):
        #permute the grouping labels
        labels = np.random.permutation(y)
        
        #difference in quantiles
        cond_q = np.percentile(x[labels == 0], 100 * q)
        exp_q  = np.percentile(x[labels == 1], 100 * q)
        
        #add to the list of sampled differences
        sample_diffs.append(exp_q - cond_q)
    
    #observed statistic for the difference
    cond_q = np.percentile(x[y == 0], 100 * q)
    exp_q  = np.percentile(x[y == 1], 100 * q)
    obs_diff = exp_q - cond_q
    
    #p-value for the result
    if alternative == 'less':
        hits = (sample_diffs <= obs_diff).sum()
    elif alternative == 'greater':
        hits = (sample_diffs >= obs_diff).sum()
    
    p_value = hits / num_trials
    end = time()
    
    if verbose:
        print('estimated p-value for the test: {:.4f} ({:.2f}%)'\
              .format(p_value, p_value*100))
        print('elapsed time: {:.3f}s'.format(end-begin))
    
    return p_value

#########1#########2#########3#########4#########5#########6#########7#########8
def fn_ranked_sum(x, 
                  y,
                  z=pd.Series([]),
                  alternative='two-sided',
                  plot=False,
                  verbose=False):
    '''This function returns a p-value for a ranked-sum test. It is presumed 
    that there are no ties.
    
    Source: Udacity Data Science Course - Lesson 5 - Statistical Considerations
    into testing - Class 10 - Forth Notebook - More Non-Parametric Tests - 
    Mann-Whitney.
    
    Inputs:
      - x (mandatory) - a vector of numpy Float, as the first entry
      - y (mandatory)  - a vector of numpy Float, as the second entry
      - z (optional) - a vector dimension (data['time'], for plotting the graph
        - (default=empty dataset) - you don´t need to inform, if you don´t intend
        to show the histograms graph
      - alternative (optional) - the test to be performed (options:'two-sided', 
        'less', 'greater') (default='two-sided')
      - plot (optional) - if you want to plot the distribution - (Boolean, 
        default=False)
      - verbose (optional) - if you want some verbosity in your function -
        (Boolean, default=False)
    Output:
      - an estimative for p-value for the ranked test
    '''
    if verbose:
        print('###function ranked sum started for ', alternative)
        
    begin=time()
    
    if plot:
        if verbose:
            print('first showing histogram graphs for both condition 0 and condition 1')
        if z.empty:
            if verbose:
                print('cannot display the graph, z parameter is missing')            
        else:
            plot = fn_plot(first_graph=x, 
                           second_graph=y,
                           aux=z,
                           type='2hist',
                           verbose=verbose)      
    
    #definining initial u as 0
    u = 0
    for i in x:
        wins = (i > y).sum()
        ties = (i == y).sum()
        u += wins + 0.5 * ties
    
    #computing z-score
    num_1 = x.shape[0]
    num_2 = y.shape[0]
    mean_u = num_1 * num_2 / 2
    stdev_u = np.sqrt( num_1 * num_2 * (num_1 + num_2 + 1) / 12 )
    z = (u - mean_u) / stdev_u
    
    #rules for the p-value, according to the test
    if alternative == 'two-sided':
        p_value = 2 * stats.norm.cdf(-np.abs(z))
    if alternative == 'less':
        p_value = stats.norm.cdf(z)
    elif alternative == 'greater':
        p_value = stats.norm.cdf(-z)
        
    end = time()
    
    if verbose:
        print('estimated p-value for the ranked sum test: {:.4f} ({:.2f}%)'\
              .format(p_value, p_value*100))
        print('elapsed time: {:.4f}s'.format(end-begin))
    
    return p_value

#########1#########2#########3#########4#########5#########6#########7#########8
def fn_sign_test(x, 
                 y,
                 z=pd.Series([]),
                 alternative='two-sided',
                 plot=False,
                 verbose=False):
    '''This function returns a p-value for a sign test. It is presumed
    that there are no ties.
    
    Source: Udacity Data Science Course - Lesson 4 - Statistical Considerations
    into testing - Class 10 - Forth Notebook - More Non-Parametric Tests.
    
    Inputs:
      - x (mandatory) - a vector of numpy Float, as the first entry
      - y (mandatory) - a vector of numpy Float, as the second entry
      - z (optional) - a vector dimension (data['time'], for plotting the graph
        - (default=empty dataset) - you don´t need to inform, if you don´t intend
        to show the histograms graph
      - alternative (optional, options: {'two-sided', 
        'less', 'greater'}) - the test to be performed (, default='two-sided')
        * two-sided -> to test for both tails of the normal distribution curve
        * less -> to test for the left tail of the normal distribution curve
        * greater -> to test for the right tail of the normal distribution curve
      - plot (optional) - if you want to plot the distribution - (Boolean, 
        default=False)
      - verbose (optional) - if you want some verbosity in your function -
        (Boolean, default=False)
    Output:
      - an estimative for p-value for the sign test
    '''
    if verbose:
        print('###function sign test started for', alternative)
        
    begin=time()
    
    if plot:
        if verbose:
            print('first showing signal test plots for both conditions')
        if z.empty:
            if verbose:
                print('cannot display the graph, z parameter is missing')            
        else:
            plot = fn_plot(first_graph=x,
                           second_graph=y,
                           aux=z,
                           type='stest',
                           verbose=True)      
   
    # compute parameters
    num = x.shape[0] - (x == y).sum()
    k = (x > y).sum() - (x == y).sum()

    # compute a p-value
    if alternative == 'two-sided':
        p_value = min(1, 2 * stats.binom(num, 0.5).cdf(min(k, num-k))) #cdf is cumulative distribution function
    if alternative == 'less':
        p_value = stats.binom(num, 0.5).cdf(k)
    elif alternative == 'greater':
        p_value = stats.binom(num, 0.5).cdf(num-k)

    end = time()
    
    if verbose:
        print('estimated p_value for sign test: {:.4f} ({:.2f}%)'.format(p_value, p_value*100))
        print('elapsed time: {:.4f}s'.format(end-begin))
   
    return p_value

#########1#########2#########3#########4#########5#########6#########7#########8
def fn_variant_analytic(df,
                        p=0.5,
                        value=0.0,
                        verbose=False):
    '''This function checks a variant metric by analytic approach. One example
    of a variant is if the migration to a new webpage format generates more
    sales.
    
    Source: Udacity Data Science Course - Lesson 4 - Statistical Considerations 
    into testing - Class 3 - First Notebook - Statistical Significance.
    
    Inputs:
      - df (mandatory) - dataset containing the binary data to be checked
      - p (optional) - probability (default=50%)
      - value (optional) - central value (default=0)
      - verbose (optional) - if you want some verbosity in your function -
        (Boolean, default=False)
    Output:
    '''
    if verbose:
        print ('###function for variant check started - analytic approach')
        
    begin = time()
    
    #number of trials
    num_observation = df.shape[0]
    num_condition = df.groupby('condition').size()
    if verbose:
        print('{} observations: {} on page H0 (control) and {} on page H1 (experiment)'\
              .format(num_observation, num_condition[0], num_condition[1]))

    #means on metric
    num_clicks= df[df['click'] == 1].groupby('condition').size()
    p_click = df.groupby('condition').mean()['click']
    diff = (((p_click[1] - p_click[0]) / p_click[0])*100)
    if verbose:
        print('user clicked on buy: {} ({:.1f}%) page H0 and {} ({:.1f}%) page H1'\
              .format(num_clicks[0], p_click[0]*100, num_clicks[1], p_click[1]*100))
        print('  - relative difference for page H1: {:.1f}%'.format(diff))
        
    #H0 -> trials & overall 'positive' rate under H0
    n_control = df.groupby('condition').size()[0]
    n_exper = df.groupby('condition').size()[1]
    p_null = df['click'].mean()

    #standard error
    std_error = np.sqrt(p_null * (1-p_null) * (1/n_control + 1/n_exper))
    if verbose:
        print('Standard Error: {:.1f}'.format(std_error))

    #z-score and p-value
    z_score = (p_click[1] - p_click[0]) / std_error
    if verbose:
        print('z-score: {:.4f}'.format(z_score))

    p_value = 1-stats.norm.cdf(z_score)
    
    end = time()
    
    if verbose:
        print()
        print('analytic p-value: {:.4f} ({:.2f}%)'.format(p_value, p_value*100))
        print('*a low value means that H0 don´t have a good explanatory power')
        print('elapsed time: {:.4f}s'.format(end-begin))
        
    return True
              
#########1#########2#########3#########4#########5#########6#########7#########8
def fn_variant_simulated(df,
                         p=0.5,
                         num_trials=200_000,
                         verbose=False):
    '''This function checks a variant metric by simulation approach.
    
    Source: Udacity Data Science Course - Lesson 4 - Statistical Considerations 
    into testing - Class 3 - First Notebook - Statistical Significance.
    
    Inputs:
      - df (mandatory) - dataset containing the binary data to be checked
      - p (optional) - probability (default=50%)
      - num_trials - number of trials for randomly simulating the experiment
      - verbose (optional) - if you want some verbosity in your function -
        (Boolean, default=False)
    Output:
      - It is a report function only! Returns True if everything runned well
    '''
    if verbose:
        print ('###function for variant check started - simulation approach')
        
    begin = time()
    
    #number of trials
    num_observation = df.shape[0]
    num_condition = df.groupby('condition').size()
    num_control = num_condition[0]
    num_experiment = num_condition[1]
    if verbose:
        print('{} observations: {} on page H0 (control) and {} on page H1 (experiment)'\
              .format(num_observation, num_control, num_experiment))
        
    #'positive' rate under null
    p_null = df['click'].mean()
    #if verbose:
    #   print('p-Null: {:.4f}'.format(p_null))
        
    #means on metric
    num_clicks= df[df['click'] == 1].groupby('condition').size()
    p_click = df.groupby('condition').mean()['click']
    diff = (((p_click[1] - p_click[0]) / p_click[0])*100)
    if verbose:
        print('user clicked on buy: {} ({:.1f}%) page H0 and {} ({:.1f}%) page H1'\
              .format(num_clicks[0], p_click[0]*100, num_clicks[1], p_click[1]*100))
        print('  - relative difference for page H1: {:.1f}%'.format(diff))

    #simulate outcomes under null, compare to observed outcome
    ctrl_clicks = np.random.binomial(num_control, p_null, num_trials)
    exp_clicks = np.random.binomial(num_experiment, p_null, num_trials)
    samples = exp_clicks / num_experiment - ctrl_clicks / num_control
    
    if verbose:
        print()
        print('*simulation part')
        print('simulated clicks on H0 (control): {} and on H1 (experiment): {}'\
              .format(len(ctrl_clicks), len(exp_clicks)))
        print('samples:', len(samples))
        
    #simulated p-value:
    p_value = (samples >= (p_click[1] - p_click[0])).mean()

    #if verbose:
    #    print('p-value: {:.4f} ({:.2f}%)'.format(p_value, p_value*100)) 

    end = time()
    
    if verbose:
        print()
        print('simulated p_value: {:.4f} ({:.2f}%)'\
              .format(p_value, p_value*100))
        print('*a low value means that H0 don´t have a good explanatory power')
        print('elapsed time: {:.4f}s'.format(end-begin))
        
    return True

###General purpose functions####################################################
#########1#########2#########3#########4#########5#########6#########7#########8
def fn_onsquare(x, 
                y, 
                verbose=False):
    '''This function calculates correlation based on on basis O(n^2).
    it fits for Kendall´s Tau.
    Inputs:
      - x vector for the first variable (numpy Array)
      - x vector for the second variable (numpy Array)
    Output:
      - a O(nLog(n)) computing time correlation.
    '''
    if verbose:
        print('*O(n^2) based correlation started')
    
    #initial parameters
    num = len(x) 
    sum_val = 0

    #loop calculating for mean values 
    for i, (x_i, y_i) in enumerate(zip(x, y)):        
        for j, (x_j, y_j) in enumerate(zip(x, y)):
            if i < j:
                sum_val += np.sign(x_i - x_j) * np.sign(y_i - y_j)
                        
    correlation = 2 * sum_val / (num * (num - 1))

    return correlation

#########1#########2#########3#########4#########5#########6#########7#########8
def fn_ologn(x, 
             y, 
             verbose=False):
    '''This function calculates correlation based on on basis O(nLog(n)).
    it fits for Pearson and Spearman.
    Inputs:
      - x vector for the first variable (numpy Array)
      - x vector for the second variable (numpy Array)
    Output:
      - a O(nLog(n)) computing time correlation.
    '''
    if verbose:
        print('*O(nLog(n)) based correlation started')
    
    #calculating 
    x_diff = x - np.sum(x) / len(x) #x - nean x
    y_diff = y - np.sum(y) / len(y) #y - mean y
    cov_xy = np.sum(x_diff * y_diff)
    var_x = np.sqrt(np.sum(x_diff ** 2))
    var_y = np.sqrt(np.sum(y_diff ** 2))
    
    correlation = cov_xy / (var_x * var_y)
    
    return correlation

#########1#########2#########3#########4#########5#########6#########7#########8
def fn_plot(first_graph, 
            second_graph=pd.Series([]),
            aux=pd.Series([]),
            type='none',
            verbose=False):
    '''This is a general function for plotting and embeleshing graphs. It was
    first designed fot the distribution for Power, and then generallized for
    other graphs on the scope of this project.
    
    Inputs:
      - first_graph (mandatory) - data (minimal) for the graph 
      - second_graph (optional) - (default=empty dataset)
      - aux (optional) - (default=empty dataset)
      - type (optional) - possible: {'none', 'htest', 'hist', '2hist', 'stest'}
        (default='none')
        * htest -> two plots, H0 and H1, in one axis, with legends
          x = p-values
          y = sum probabilities for each graph
          + vertical line for p_crit value for H0 explanatory power
          + fills for showing the explanatory power
          first_graph - H0 series of data (normal continous random variable)
          contains y_null for shaping H0
          second_graph - H1 series of data (normal continous random variable)
          contains y_alt for shaping H1
          aux - p_crit (1-alpha) - the critical point thal over it, H0 does not
          explain the phenomena 
        * hist -> one histogram in one axis, no legends
          x = time
          y = counts for each bin, given n bins (automatically calculated)
          first_graph - series of time-values 
          second_graph - not used
          aux - not used
        * 2hist -> two histograms (control, experiment) in one axis, with legends
          x = time
          y = counts for each bin, for two partially superposed histograms
          first_graph - control data
          second_graph - experiment data
          aux -
        * stest -> two plots (control, experiment) in one axis, with legends
          x = time (days of experiment)
          y = success rate for each experiment
          first_graph - success rate for control
          second_graph - success rate for experiment
          aux - series of days when the experiment was running
      - verbose (optional) - if you want some verbosity in your function -
        (Boolean, default=False)
    Output - True, if everything goes well - this is a plot function only!
    '''
    if verbose:
        print('###function plot started')        
    
    #making the plot
    mstyles.use('seaborn-darkgrid')#ggplot') #dark_background')
    fig_zgen = mpyplots.figure() #creating the object    
    axis_zgen = fig_zgen.add_axes([0,0,1,1]) #X0 y0 width height
    
    if type == 'htest': #histogram for h0 h1 test
        #assertions
        #assert second_graph exists
        assert aux > 0. #aux receives p_critical
        if verbose:
            print('*plotting hypothesis test')
        #preprocessing
        low_bound = first_graph.ppf(.01) #null hypothesis distribution
        high_bound = second_graph.ppf(.99) #alternative hypothesis distribution
        x = np.linspace(low_bound, high_bound, 201)
        y_null = first_graph.pdf(x) #null
        y_alt = second_graph.pdf(x) #alternative
        #plotting
        axis_zgen.plot(x, y_null)
        axis_zgen.plot(x, y_alt)
        axis_zgen.vlines(aux, 
                         0, 
                         np.amax([first_graph.pdf(aux), second_graph.pdf(aux)]),
                         linestyles = '--', color='red')
        axis_zgen.fill_between(x, y_null, 0, where = (x >= aux), alpha=0.5)
        axis_zgen.fill_between(x, y_alt , 0, where = (x <= aux), alpha=0.5)
        axis_zgen.legend(labels=['null hypothesis','alternative hypothesis'], fontsize=12)
        title = 'Hypothesis Test'
        x_label = 'difference'
        y_label = 'density'
        
    elif type == 'hist': #time count histogram
        #assertions
        #asserts second graph is False
        #asserts aux is False
        if verbose:
            print('*showing data histogram')
        n_bins = np.arange(0, first_graph.max()+400, 400)
        mpyplots.hist(first_graph, 
                      bins = n_bins)
        title = 'Time Histogram'
        x_label = 'time'
        y_label = 'counts'
    
    elif type == '2hist':
        #assertions
        #assert second_graph
        #assert aux == data['time'] #aux receives data['time']
        counts1 = first_graph
        counts2 = second_graph
        if verbose:
            print('*showing test (control and experiment) histograms')
        #plotting
        borders = np.arange(0, aux.max()+400, 400)
        mpyplots.hist(counts1, alpha=0.5, bins=borders)
        mpyplots.hist(counts2, alpha=0.5, bins=borders)
        axis_zgen.legend(labels=['control', 'experiment'], fontsize=12)
        title = 'Time Histogram'
        x_label = 'time'
        y_label = 'counts'
        
    elif type == 'stest':
        #assertions
        #assert second_graph
        #assert aux == data['day'] 
        if verbose:
            print('*plotting signal test (control and experiment) graphs')
        #preprocessing
        x=aux
        y_control=first_graph
        y_experiment=second_graph
        #plotting
        axis_zgen.plot(x, y_control)
        axis_zgen.plot(x, y_experiment)      
        axis_zgen.legend(labels=['control', 'experiment'], fontsize=12)
        title = 'Signal Test'
        x_label = 'day of experiment'
        y_label = 'success rate'
        
    elif type == 'none':
        if verbose:
            print('*no graph type has been choosen, nothing was plotted')
        return False
       
    else:
        raise Exception('type of graph invalid or not informed')
    
    fig_zgen.suptitle(title, fontsize=14, fontweight='bold')
    mpyplots.xlabel(x_label, fontsize=14)
    mpyplots.ylabel(y_label, fontsize=14)
    mpyplots.show()
    
    return True

#########1#########2#########3#########4#########5#########6#########7#########8
def fn_read_data(filepath,
                 index=False,
                 index_col='id',
                 remove_noisy_cols=False,
                 verbose=False):
    '''This function reads a .csv file. It was first designed for the first
    project of this course and then generallized for other imports. The idea
    is to make the first dataset testing and modification on it.
    
    Inputs:
      - filepath (mandatory) - String containing the full path for the data to
        oppened
      - index_col (optional) - String containing the name of the index column
        (default='id')
      - remove_noisy_cols - if you want to remove noisy (blank) columns
        (default=False)
      - verbose (optional) - if you needed some verbosity, turn it on - Boolean 
        (default=False)
    Output:
      - Pandas Dataframe with the data
    *this function came from my library udacourse2.py and was adapted for this
    project
    '''
    if verbose:
        print('###function read data from .csv file started')
        
    begin=time()
    
    #reading the file
    df = pd.read_csv(filepath)
    if remove_noisy_cols:
        del df['Unnamed: 0']
    if index:
        df.set_index(index_col)
    
    if verbose:
        print('file readed as Dataframe')

    #testing if Dataframe exists
    #https://stackoverflow.com/questions/39337115/testing-if-a-pandas-dataframe-exists/39338381
    if df is not None: 
        if verbose:
            print('dataframe created from', filepath)
            #print(df.head(5))
    else:
        raise Exception('something went wrong when acessing .csv file', filepath)
    
    #setting a name for the dataframe (I will cound need to use it later!)
    ###https://stackoverflow.com/questions/18022845/pandas-index-column-title-or-name?rq=1
    #last_one = filepath.rfind('/')
    #if last_one == -1: #cut only .csv extension
    #    df_name = filepath[: -4] 
    #else: #cut both tails
    #    df_name = full_path[last_one+1: -4]   
    #df.index.name = df_name
    #if verbose:
    #    print('dataframe index name set as', df_name)
              
    end = time()
    
    if verbose:
        print('elapsed time: {:.4f}s'.format(end-begin))
              
    return df

###DEPRECATED###################################################################
#########1#########2#########3#########4#########5#########6#########7#########8
def fn_movie_liked(item, 
                   min_rating=7,
                   sort=True,
                   verbose=False):
    '''This function takes all the items for one user and return the best rated
    items.
    
    Source: Udacity Data Science Course - Lesson 6  - Identifying Reccomendations
    Forth Notebook - Class 17 - Collaborative Filtering - Recommendations with 
    MovieTweetings - Collaborative Filtering

    Inputs:
      - item (mandatory) - a dataset filtered for an individual user.
        e.g. (for user 66): user_item['user_id'] == 66  
      - min_rating (optional) - the trigger point to consider an item to be 
        considered "nice" for an user (integer, default=7)
      - sort (optional) - if you want to show the most rated items first
        (Boolean, default=True)
      - verbose (optional) - if you want some verbosity in your function -
        (Boolean, default=False)
    Output:
      - movie_liked - an array of movies the user has watched and liked
    '''
    raise Exception('function deprecated, use fn_movie_liked2')
    
    try:
        user = item.iloc[0]
    except TypeError:
        if verbose:
            print('you must inform a filtered dataset, see documentation')
        return False
    except IndexError:
        if verbose:
            print('an empty dataset was informed, this user does not exist')
        return False

    if verbose:
        print('###function movies liked started for user', user)

    begin = time()
    movie_liked = item[item['rating'] > 7]
    
    if sort:
        if verbose:
            print('*sorting the list - best rated first')
        movie_liked = movie_liked.sort_values(by='rating', ascending=False)

    movie_liked = np.array(movie_liked['movie_id'])    
    end = time()
    
    if verbose:
        print('elapsed time: {:.4f}s'.format(end-begin))
    
    return movie_liked

#########1#########2#########3#########4#########5#########6#########7#########8
#for reloading this library on Jupyter, just run this lines in a code cell:
#from importlib import reload 
#import udacourse3
#udacourse3 = reload(udacourse3)