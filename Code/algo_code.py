import numpy as np
import statsmodels.api as sm
import pandas as pd

import quantopian.optimize as opt
import quantopian.algorithm as algo

import sklearn as sk
import sklearn.cluster as skc
import sklearn.decomposition as skd
import sklearn.manifold as skm
import statsmodels.tsa.stattools as stattools

from scipy import stats

from quantopian.pipeline.data import morningstar
from quantopian.pipeline.filters.morningstar import Q500US, Q1500US, Q3000US
from quantopian.pipeline import Pipeline
from quantopian.algorithm import attach_pipeline, pipeline_output 
from quantopian.pipeline.filters import QTradableStocksUS

# Use a random forest classifier. 
# More here: http://scikit-learn.org/stable/user_guide.html
from sklearn import linear_model, metrics
from sklearn.cross_validation import train_test_split
from sklearn.neural_network import BernoulliRBM
import sklearn.pipeline as skp

# Global Constant --------------------------------------------------------------
TRADE_UNIQUE = False # Only trade unique pairs with unique stocks in portfolio
WITH_CLASSIFIER = True # Choose to trade with the classfier
STOP_LOSS_2 = True # include stop-loss with 2-touch barrier
STOP_LOSS_4 = False # include stop-loss with 4-touch barrier
TRADE_HEDGED = True # Trade completely hedged
OR_SIGNAL = False # Classification: x_signal or/and y_signal
NUM_SHARES = 1 # Number of shares of stock Y to be bought and hedged

def initialize(context): # -----------------------------------------------------
    # Quantopian backtester specific variables
    set_slippage(slippage.FixedSlippage(spread=0))
    set_commission(commission.PerShare(cost=0.001, min_trade_cost=1))
    
    attach_pipeline(make_pipeline(), 'my_pipeline')
    
    context.stock_pairs = dict()

    context.stocks = set()
    
    context.num_pairs = 10
    
    # strategy specific variables
    context.lookback = 20 # used for regression
    context.z_window = 20 # used for zscore calculation, must be <= lookback
    context.num_std = 2 # number of standard deviations above or below the mean
    
    context.target_weights = pd.Series(index=context.stocks, data=0)
    
    # Only do work 30 minutes before close
    schedule_function(func=check_pair_status, 
                      date_rule=date_rules.every_day(),
                      time_rule=time_rules.market_close(minutes=30))

    schedule_function(func=update_pairs, 
                      date_rule=date_rules.week_start(days_offset=1),
                      time_rule=time_rules.market_close(minutes=30))
                      
# Will be called on every trade event for the securities you specify.
def handle_data(context, data): # ----------------------------------------------
    # Our work is now scheduled in check_pair_status
    pass


def update_pairs(context, data): # ---------------------------------------------
    # Our work is now scheduled in check_pair_status
    pairs = get_stock_pairs(context, data)
    
    new_stocks = set()
    # Add pairs if they are not already contained in our dictionary
    for pair in pairs:
        (stock1, stock2) = pair
        if TRADE_UNIQUE:
            if (pair not in context.stock_pairs) and \
            ((stock1 not in context.stocks) and (stock2 not in context.stocks)):
                context.stock_pairs[pair] = {'spread': np.empty((1,0), float), 
                                             'inLong': False, 'inShort': False, 
                                             "low_touch": 0, "upp_touch": 0}
                
                # Add new stocks
                context.stocks.add(stock1)
                new_stocks.add(stock1)
                context.stocks.add(stock2)
                new_stocks.add(stock2)
                
                compute_prev_spreads(pair, context, data)
        else: 
            if (pair not in context.stock_pairs) and \
            ((stock1 not in context.stocks) or (stock2 not in context.stocks)):
                context.stock_pairs[pair] = {'spread': np.empty((1,0), float), 
                                             'inLong': False, 'inShort': False, 
                                             "low_touch": 0, "upp_touch": 0}
                
                # Add new stocks
                if (stock1 not in context.stocks):
                    context.stocks.add(stock1)
                    new_stocks.add(stock1)
                if (stock2 not in context.stocks):
                    context.stocks.add(stock2)
                    new_stocks.add(stock2)
                    
                compute_prev_spreads(pair, context, data)
    
    # strategy specific variables
    if len(new_stocks) != 0:
        new_stock_weights = pd.Series(index=list(new_stocks), data=0.0)
        B = new_stock_weights
        A = context.target_weights
        context.target_weights = pd.concat([A, B[B.index.difference(A.index)]])
    
def check_pair_status(context, data): # ---------------------------------------------------
    
    prices = data.history(list(context.stocks), 'price', 
        35, '1d').iloc[-context.lookback::]
    prices = prices.fillna(method='ffill')
    
    stock_pairs_dict = context.stock_pairs.copy()
    
    for pair, params in stock_pairs_dict.items():

        (stock_y, stock_x) = pair

        try: 
            Y = prices[stock_y]
            X = prices[stock_x]
        except:
            continue
        
        positions = context.portfolio.positions
        num_positions = len(context.portfolio.positions)
        
        # # Try to compute the hedge ratio by OLS, otherwise go to next pair
        try:
            hedge = hedge_ratio(np.log(Y), np.log(X), add_const=True)      
        except:
            continue

        context.target_weights = get_current_portfolio_weights(context, data)
        
        # Get the new spread and append to our history of spread for our current 
        # stock pair
        new_spread = np.log(Y[-1]) - hedge * np.log(X[-1])
        context.stock_pairs[pair]['spread'] = np.append(params['spread'], 
            np.array([[new_spread]]), axis = 1)
        
        # Get rid of pair that is more than 100 trading days old
        if params['spread'].shape[1] >= 100:
            liquidate(pair, context, data)
            continue 
        
        # Begin Trade of we have at least z_window amount of data
        if params['spread'].shape[1] > context.z_window:
            # Keep only the z-score lookback period
            spreads = params['spread'][:, -context.z_window:]

            zscore = (spreads[:, -1] - spreads.mean()) / spreads.std()

            if params['inShort'] and zscore < 0.0:
                liquidate(pair, context, data)
                continue

            if params['inLong'] and zscore > 0.0:
                liquidate(pair, context, data)
                continue
            
            # Double Touch Barrier with Stop Loss (Lower Barrier)
            if zscore < -context.num_std and num_positions <= context.num_pairs:
                if (not params['inLong']) and params['low_touch'] == 0:
                    params['low_touch'] = 1
                elif params['low_touch'] == 1:
                    continue
                elif STOP_LOSS_4 and params['low_touch'] == 2:
                    params['low_touch'] = 3
                else:
                    liquidate(pair, context, data)
                continue
                
            if zscore > -context.num_std and num_positions <= context.num_pairs:
                if STOP_LOSS_4:
                    if params['low_touch'] == 3:
                        params['low_touch'] = 4
                if not TRADE_UNIQUE:
                    if (stock_y in positions) or (stock_x  in positions):
                        continue
                if (not params['inLong']) and params['low_touch'] == 1:
                    if WITH_CLASSIFIER:
                        y_signal = get_price_signal(stock_y, data)
                        x_signal = get_price_signal(stock_x, data)
                    else:
                        y_signal = 1
                        x_signal = 0
                    
                    if OR_SIGNAL:
                        combined_signal = y_signal == 1 or x_signal == 0
                    else:
                        combined_signal = y_signal == 1 and x_signal == 0
                    
                    if combined_signal:
                        if STOP_LOSS_2:
                            params['low_touch'] = 2
                
                        # Only trade if NOT already in a trade 
                        y_target_shares = NUM_SHARES
                        X_target_shares = -hedge
                        if TRADE_HEDGED:
                            X_target_shares = (-Y[-1] * y_target_shares) / X[-1]
                        context.stock_pairs[pair]['inLong'] = True
                        context.stock_pairs[pair]['inShort'] = False

                        (y_target_pct, x_target_pct) = \
                        computeHoldingsPct(y_target_shares, X_target_shares, 
                                           Y[-1], X[-1])
            
                        context.target_weights[stock_y] = \
                        y_target_pct *(1.0 / context.num_pairs)
                        context.target_weights[stock_x] = \
                        x_target_pct *(1.0 / context.num_pairs)
            
                        # Replace any NA values with 0
                        context.target_weights = context.target_weights.fillna(0)
            
                        record(Y_pct=y_target_pct, X_pct=x_target_pct)
                        allocate(context, data)
                        continue
                    else:
                        params['low_touch'] = 0
                        continue
                        
            # Double Touch Barrier with Stop Loss (Upper Barrier)
            if zscore > context.num_std and num_positions <= context.num_pairs:
                if (not params['inShort']) and params['upp_touch'] == 0:
                    params['upp_touch'] = 1
                elif params['upp_touch'] == 1:
                    continue
                elif STOP_LOSS_4 and params['upp_touch'] == 2:
                    params['upp_touch'] = 3
                else:
                    liquidate(pair, context, data)
                continue
            
            if zscore < context.num_std and num_positions <= context.num_pairs:
                if STOP_LOSS_4:
                    if params['upp_touch'] == 3:
                        params['upp_touch'] = 4
                if not TRADE_UNIQUE:
                    if (stock_y in positions) or (stock_x  in positions):
                        continue
                if (not params['inShort']) and params['upp_touch'] == 1:
                    if WITH_CLASSIFIER:
                        y_signal = get_price_signal(stock_y, data)
                        x_signal = get_price_signal(stock_x, data)
                    else:
                        y_signal = 0
                        x_signal = 1
                    
                    if OR_SIGNAL:
                        combined_signal = y_signal == 0 or x_signal == 1
                    else:
                        combined_signal = y_signal == 0 and x_signal == 1
                    
                    if combined_signal:
                        if STOP_LOSS_2:
                            params['upp_touch'] = 2
                        y_target_shares = -NUM_SHARES
                        X_target_shares = hedge
                        if TRADE_HEDGED:
                            X_target_shares = (-Y[-1] * y_target_shares) / X[-1]
                        context.stock_pairs[pair]['inShort'] = True
                        context.stock_pairs[pair]['inLong'] = False

                        (y_target_pct, x_target_pct) = \
                        computeHoldingsPct(y_target_shares, X_target_shares, 
                                           Y[-1], X[-1] )
                        context.target_weights[stock_y] = \
                        y_target_pct * (1.0/context.num_pairs)
                        context.target_weights[stock_x] = \
                        x_target_pct * (1.0/context.num_pairs)
            
                        # Replace any NA values with 0
                        context.target_weights = context.target_weights.fillna(0)
            
                        record(Y_pct=y_target_pct, X_pct=x_target_pct)
                        allocate(context, data)
                        continue
                    else:
                        params['upp_touch'] = 0
                        continue

def hedge_ratio(Y, X, add_const=True): # ---------------------------------------
    if add_const:
        X = sm.add_constant(X)
        model = sm.OLS(Y, X).fit()
        return model.params[1]
    model = sm.OLS(Y, X).fit()
    return model.params.values
   
def computeHoldingsPct(yShares, xShares, yPrice, xPrice):# ---------------------
    yDol = yShares * yPrice
    xDol = xShares * xPrice
    notionalDol =  abs(yDol) + abs(xDol)
    y_target_pct = yDol / notionalDol
    x_target_pct = xDol / notionalDol
    return (y_target_pct, x_target_pct)

def get_current_portfolio_weights(context, data):# ----------------------------- 
    positions = context.portfolio.positions  
    positions_index = pd.Index(positions)  
    share_counts = pd.Series(  
        index=positions_index,  
        data=[positions[asset].amount for asset in positions]  
    )

    current_prices = data.current(positions_index, 'price')  
    current_weights = \
    share_counts * current_prices / context.portfolio.portfolio_value  
    return current_weights.reindex(positions_index.union(context.stocks), 
        fill_value=0.0)

def allocate(context, data): # -------------------------------------------------   
    # Set objective to match target weights as closely as possible, 
    # given constraints
    objective = opt.TargetWeights(context.target_weights)
    
    # Define constraints
    constraints = []
    constraints.append(opt.MaxGrossExposure(1.0))
    
    algo.order_optimal_portfolio(
        objective=objective,
        constraints=constraints,
    )

def make_pipeline(): # ---------------------------------------------------------
    # Define Universe 
    universe = QTradableStocksUS()

    # Get Fiundamental Data
    pipe = Pipeline(
        columns= {
            'Market Cap': morningstar.valuation.market_cap.latest.quantiles(5),
            'Industry': \
            morningstar.asset_classification.morningstar_industry_group_code.latest,
            'Financial Health': \
            morningstar.asset_classification.financial_health_grade.latest,
            'book_value_per_share': \
            morningstar.valuation_ratios.book_value_per_share.latest,
            'value_score': morningstar.Fundamentals.value_score.latest,
            'net_margin': morningstar.operation_ratios.net_margin.latest,
            'sector': morningstar.Fundamentals.morningstar_sector_code.latest
        },
        screen=universe
    )
    return pipe
    
    
def get_stock_pairs(context, data): # ------------------------------------------
    N_PRIN_COMPONENTS = 50
    sector_map = {
    101: "Basic_Materials",
    102: "Consumer_Cyclical",
    103: "Financial_Services",
    104: "Real_Estate",
    205: "Consumer_Defensive",
    206: "Healthcare",
    207: "Utilities",
    308: "Communication_Services",
    309: "Energy",
    310: "Industrials",
    311: "Technology"}

    res = pipeline_output('my_pipeline')

    res.dropna(inplace=True)
    
    # remove stocks in Industry "Conglomerates"
    res = res[res['Industry']!=31055]
    
    res["sector"] = res.sector.apply(lambda x: sector_map[x])
    
    # replace the categorical data with numerical scores per the docs
    res['Financial Health'] = res['Financial Health'].astype('object')
    health_dict = {u'A': 0.1,
                   u'B': 0.3,
                   u'C': 0.7,
                   u'D': 0.9,
                   u'F': 1.0}
    res = res.replace({'Financial Health': health_dict})
    
    # Get Pricing Data and compute returns
    pricing = data.history(res.index, 'price', bar_count=100, frequency='1d')

    returns = pricing.pct_change()

    # we can only work with stocks that have the full return series
    returns = returns.iloc[1:,:].dropna(axis=1)

    # Perform PCA
    pca = skd.PCA(n_components= N_PRIN_COMPONENTS)
    pca.fit(returns)

    res = pd.get_dummies(res)

    cols = []
    for key in sector_map:
        cols.append("sector_" + sector_map[key])

    X = np.hstack(
        (pca.components_.T,
         res['Market Cap'][returns.columns].values[:, np.newaxis],
         res['Financial Health'][returns.columns].values[:, np.newaxis],
         res['book_value_per_share'][returns.columns].values[:, np.newaxis],
         res['net_margin'][returns.columns].values[:, np.newaxis],
         res['value_score'][returns.columns].values[:, np.newaxis])
    )

    for col in cols:
        X = np.hstack((X, res[col][returns.columns].values[:, np.newaxis]))

    # Sphere the data (x - mu) / s
    X = sk.preprocessing.StandardScaler().fit_transform(X)
    
    # Perform clustering
    clf = skc.DBSCAN(eps=1.9, min_samples=3)

    clf.fit(X)
    clustered = clf.labels_

    clustered_series = pd.Series(index=returns.columns, data=clustered.flatten())
    clustered_series = clustered_series[clustered_series != -1]

    CLUSTER_SIZE_LIMIT = 9999
    counts = clustered_series.value_counts()
    ticker_count_reduced = counts[(counts>1) & (counts<=CLUSTER_SIZE_LIMIT)]

    cluster_dict = {}
    for i, which_clust in enumerate(ticker_count_reduced.index):
        tickers = clustered_series[clustered_series == which_clust].index
        score_matrix, pvalue_matrix, pairs = find_cointegrated_pairs(
            pricing[tickers]
        )
        cluster_dict[which_clust] = {}
        cluster_dict[which_clust]['score_matrix'] = score_matrix
        cluster_dict[which_clust]['pvalue_matrix'] = pvalue_matrix
        cluster_dict[which_clust]['pairs'] = pairs

    pairs = []
    for clust in cluster_dict.keys():
        pairs.extend(cluster_dict[clust]['pairs'])
    
    d = sorted(pairs, key=lambda d: list(d.values()))
    d = [tuple(dict_i.keys())[0] for dict_i in d]
    return d


def find_cointegrated_pairs(stock_data, significance=0.05): # ------------------
    # This function is from, 
    # https://www.quantopian.com/lectures/introduction-to-pairs-trading. 
    # We also sort the pairs in ascending order by p-value
    n = stock_data.shape[1]
    score_matrix = np.zeros((n, n))
    pvalue_matrix = np.ones((n, n))
    keys = stock_data.keys()
    pairs = []
    for i in range(n):
        for j in range(i+1, n):
            S1 = stock_data[keys[i]]
            S2 = stock_data[keys[j]]
            result = stattools.coint(S1, S2)
            score = result[0]
            pvalue = result[1]
            score_matrix[i, j] = score
            pvalue_matrix[i, j] = pvalue
            if pvalue < significance:
                pairs.append({(keys[i], keys[j]): pvalue})
    return score_matrix, pvalue_matrix, pairs


def stock_in_dict(stock, context, data): # -------------------------------------
    # Returns true if the stock is contained in the stock pairs dictionary
    for pair in context.stock_pairs:
        if stock in pair:
            return True
    return False

def liquidate(pair, context, data): # ------------------------------------------
    # Close our postion in pair
    (stock_y, stock_x) = pair
    context.target_weights[stock_y] = 0.0
    context.target_weights[stock_x] = 0.0
    
    context.stock_pairs[pair]['inLong'] = False
    context.stock_pairs[pair]['inShort'] = False
    
    del context.stock_pairs[pair]
    record(X_pct=0, Y_pct=0)
    context.target_weights = context.target_weights.fillna(0)
    allocate(context, data)
                
    # No longer need to track target weights
    if not stock_in_dict(stock_y, context, data):
        context.target_weights = context.target_weights.drop([stock_y])
        context.stocks.remove(stock_y)
    if not stock_in_dict(stock_x, context, data):
        context.target_weights = context.target_weights.drop([stock_x])
        context.stocks.remove(stock_x)
        
def get_price_signal(stock, data): # -------------------------------------------
    # Use Logistic Regression classifier with BernoulliRBM Neural Netowrk
    # Return 1 if buy signal on stock return, Return 0 if sell signal
    price = data.history(stock, 'price', bar_count=50, frequency='1d')
    price = price.fillna(method='ffill')

    logistic = linear_model.LogisticRegression()
    rbm = BernoulliRBM(random_state=0, verbose=False)

    rbm.learning_rate = 0.017
    rbm.n_iter = 30

    rbm.n_components = 150
    logistic.C = 6000.0
    classifier = skp.Pipeline(steps=[('rbm', rbm), ('logistic', logistic)])
    
    # Make a list of 1's and 0's, 1 when the price increased from the prior bar
    returns = np.diff(price)
    changes = (np.diff(price) > 0).astype(int)
    
    lag = 1
    X = (returns)[:-lag].astype(float) # Add the prior changes
    X_data = X.reshape((len(X), 1))
    Y = changes[lag:] # Add dependent variable, the final change
    Y_data = Y.reshape((len(Y), 1))

    
    if len(Y) >= 30: # There needs to be enough data points to make a good model
        try:
            classifier.fit(X_data, Y_data) # Generate the model
            prediction = classifier.predict(returns[-lag:]) # Predict
        except:
            return None
        return prediction[-1]

def compute_prev_spreads(pair, context, data): # -------------------------------
    # Fill historical spread for a stock pair
    for n in range(context.lookback, 0, -1):
        prices = data.history(list(context.stocks), 'price', 
            80, '1d').iloc[-context.lookback-n:-n:]
        prices = prices.fillna(method='ffill')
    
        params = context.stock_pairs[pair]
        
        (stock_y, stock_x) = pair

        try: 
            Y = np.log(prices[stock_y])
            X = np.log(prices[stock_x])
        except:
            continue
       
        # Try to compute the hedge ratio by OLS, otherwise go to next pair
        try:
            hedge = hedge_ratio(Y, X, add_const=True)      
        except:
            continue
        # Get the new spread and append to our history of spread for our 
        # current stock pair
        new_spread = Y[-1] - hedge * X[-1]
        context.stock_pairs[pair]['spread'] = np.append(params['spread'], 
            np.array([[new_spread]]), axis = 1)
        
        
        
        