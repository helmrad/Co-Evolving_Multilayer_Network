import pandas as pd
import numpy as np
import requests
import time
import io
import datetime
import parameters as pars


def retrieve():
    # Get trade data of European countries from UN's COMTRADE
    # Get list of countries
    data = pd.read_excel('data/countries.xlsx', nrows=pars.nodes)
    countries = list(data['Country'].values)
    # Retrieve country codes
    reporters_url = 'https://comtrade.un.org/Data/cache/reporterAreas.json'
    partners_url = 'https://comtrade.un.org/Data/cache/partnerAreas.json'
    reporters_raw = requests.get(reporters_url).json()['results']
    partners_raw = requests.get(partners_url).json()['results']
    # Build dictionaries that map countries to Comtrade's code system
    reporters = {}
    for entry in reporters_raw:
        reporters[entry['text']] = entry['id']
    partners = {}
    for entry in partners_raw:
        partners[entry['text']] = entry['id']

    present = datetime.date.today().strftime('%Y%m%d')
    # Define a time span of interest
    years = np.arange(pars.years[0], pars.years[1])
    # Decompose the url that will be used for data requests
    url_a = 'https://comtrade.un.org/api/get?r='  # insert reporter
    url_b = '&px=HS&ps=ALL&p='  # insert partner
    url_c = '&rg='  # specify trade direction: 1/2 for imports/exports
    url_d = '&cc=TOTAL&fmt=csv&freq=M'
    # Initialize tensor to hold data
    trade_data = np.zeros((len(years)*pars.months, 2, len(countries), len(countries)))
    # Initialize counter for COMTRADE compliance
    cnt = 0
    for c_a in range(0, len(countries)):
        for c_b in range(0, len(countries)):
            if countries[c_a] != countries[c_b]:
                for i in [0, 1]:
                    # Build the url to request data
                    url = url_a + reporters[countries[c_a]] + url_b + partners[countries[c_b]] + url_c + str(i+1) + url_d
                    # Try to fetch data until success
                    while True:
                        try:
                            un_data = requests.get(url)
                            status_code = un_data.status_code
                            cnt += 1
                        except:
                            status_code = 409
                        if status_code != 409:  # if extraction was successful
                            # Convert the data into a DataFrame
                            df = pd.read_csv(io.StringIO(un_data.content.decode('utf-8')))
                            # Make sure data is sorted chronologically
                            df = df.sort_values(['Period']).reset_index(drop=True)
                            # Get initial year and month of the data
                            year_a, mo_a = df['Year'].to_numpy()[0], df['Period'].to_numpy()[0]%100  # XXXXYY mod 100 to get YY
                            # Compute indices for placing the data into the tensor
                            ind_a = (year_a - pars.years[0]) * pars.months + mo_a - 1  # -1 since pythonic indices start at 0
                            # Get trade data
                            trade = df['Trade Value (US$)'].to_numpy()
                            try:  # check if data is corrupted
                                a = trade_data[ind_a:ind_a+len(trade), :, :, :]
                                break
                            except:
                                print('Got invalid value in double_scalars error')
                        else:  # data retrieval was unsuccessful
                            print('Got 409')
                    # Assign the retrieved data to the interaction tensor
                    if i == 0:  # imports
                        trade_data[ind_a:ind_a+len(trade), i, c_b, c_a] = list(trade)
                    else:  # exports
                        trade_data[ind_a:ind_a+len(trade), i, c_a, c_b] = list(trade)
                    # Interleave one second break in between requests in compliance with COMTRADE regulations
                    time.sleep(1)
                print('Retrieved trade data between ' + countries[c_a] + ' and ' + countries[c_b])
                # Check whether usage limit is reached
                if cnt >= 98:
                    print('100 requests reached at ' + datetime.datetime.now().strftime("%H:%M:%S"))
                    # Interleave one hour break in compliance with COMTRADE regulations
                    cnt = 0
                    time.sleep(4000)
    # Save data
    np.save('data/' + present + '_trade_data', trade_data)


def merge_datasets():
    # Get list of countries
    data = pd.read_excel('data/countries.xlsx', nrows=pars.nodes)
    countries = list(data['Country'].values)
    years = np.arange(pars.years[0], pars.years[1])
    trade_data = np.zeros((len(years)*pars.months, 2, len(countries), len(countries)))
    # Data paths
    paths = ['20200709_trade_dataFranceBulgaria.npy',
             '20200712_trade_data_NetherlandsSweden.npy',
             '20200713_trade_data_CzechiaBulgaria.npy']
    indices = [0, 3, 10, 13]
    # Merge
    for p, i in zip(paths, range(0, len(indices)-1)):
        trade_data[:, :, indices[i]:indices[i+1], :] = np.load('data/' + p)[:, :, indices[i]:indices[i+1], :]

    present = datetime.date.today().strftime('%Y%m%d')
    np.save('data/' + present + '_trade_data_merge', trade_data)


if __name__ == "__main__":
    #merge_datasets()
    retrieve()
