import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_absolute_error, mean_squared_error
import streamlit as st
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from prophet import Prophet
import ollama
from sklearn.decomposition import PCA
import plotly.express as px
import os 
from openai import OpenAI

st.set_page_config(
    page_title="World Trends Analyzer",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded")

api_key = st.secrets["GITHUB_TOKEN"]
endpoint = "https://models.github.ai/inference"
model = "openai/gpt-4.1-mini"



# alt.themes.enable("dark")







st.title("World Trends Analyzer")

df=pd.read_csv("master_dataset_again.csv")
com=pd.read_csv("HDI_Comparison.csv")

df2=pd.read_csv("Renewable_energy_consumption.csv")

df2=df2[['Country','Code']]

df2 = df2.drop_duplicates(subset=['Code'])

m=df.merge(df2,on='Country',how="left")

tab1, tab2, tab3, tab4,tab5,tab6 = st.tabs(["Overview", "Detailed Analysis", "Country Profiles","Comparison","Future Predictions","AI"])

with tab1:
    st.title("🌍 Dashboard Overview")
    def Global_map(df):
 

        g = st.selectbox("Choose : ", [
            'Life expectancy at birth, total (years)',
            'Literacy rate, adult total (% of people ages 15 and above)',
            'Renewable energy consumption', 'GDP per capita (current US$)',
            'Annual CO₂ emissions (per capita)',
            'Child mortality rate of children aged under five years, per 100 live births',"OBS_VALUE:Observation Value"])

        y = st.selectbox("Year: ", [1990, 1991, 1992, 1993, 1994, 1995, 1996, 1997, 1998, 1999, 2000,
            2001, 2002, 2003, 2004, 2005, 2006, 2007, 2008, 2009, 2010, 2011,
            2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022])

        m_year = m[m['Year']==y]
        import plotly.express as px

        fig = px.choropleth(
            m_year,
            locations="Code",                 
            color=g,
            hover_name="Country",             
            color_continuous_scale="YlGnBu",
            title=f"Global {g}"
        )
        st.plotly_chart(fig, use_container_width=True)

        
    Global_map(df)
    # fig.show()



    col = st.columns((4, 5, 5), gap='medium')

    with col[0]:
        avg_gdp = df['GDP per capita (current US$)'].mean()
        literacy = df['Literacy rate, adult total (% of people ages 15 and above)'].mean()
        countries=df['Country'].unique().size
        # Show metrics
        st.metric("Total Number of Countries",f"{countries}")
        st.metric("Global Average Gross Domestic Product per capita 1990-2022 ", f"{avg_gdp:.2f}$")
        st.metric("Global Average Literacy Rate 1990-2022 ", f"{literacy:.1f}%")
        # st.metric("💨 Average", f"{total_co2:,.0f} Mt")

        


    with col[1]:

        avg_hdi = com["Development_Index"].mean()
        renewable_usage = df['Renewable energy consumption'].mean()
        total_co2 = df["Annual CO₂ emissions (per capita)"].sum()

        # Show metrics
        st.metric("Global Average HDI 1990-2022", f"{avg_hdi:.2f}")
        st.metric("Global Avergae Renewable Energy Consumption 1990-2022 ", f"{renewable_usage:.1f}%")
        st.metric("Global Total CO₂ Emissions 1990-2022", f"{total_co2:,.0f} tons")

    with col[2]:

        top5 = com.sort_values("Development_Index", ascending=False).head(5)
        bottom5 = com.sort_values("Development_Index", ascending=True).head(5)

        if "show_top" not in st.session_state:
                st.session_state.show_top = True

        if st.button("🔄 Toggle Top/Bottom 5"):
                st.session_state.show_top = not st.session_state.show_top

            # Show table
        if st.session_state.show_top:
                st.write("🔝 Top 5 Countries by HDI")
                st.table(top5[["Country", "Development_Index"]])
        else:
                
                st.write("⬇️ Bottom 5 Countries by HDI")
                st.table(bottom5[["Country", "Development_Index"]])



with tab2:
    st.title("📊 Detailed Analysis")
    choice = st.selectbox("Choose chart: ", ["Heatmap","Yearly Country-wise Trends","Country-wise Top and Bottom Trends (Top 5)","Vaccine with Most Coverage","Yearly Global Trends","Country-wise Average Trends (All)","Country-wise Metrics Distributions","Yearly Metric Comparison","Clusters"],index=None,placeholder="Select")

    def plot_yearly_country_trend(df):

        choice = st.selectbox("Choose value: ", [
        'Life expectancy at birth, total (years)',
        'Literacy rate, adult total (% of people ages 15 and above)',
        'Renewable energy consumption', 'GDP per capita (current US$)',
        'Annual CO₂ emissions (per capita)',
        'Child mortality rate of children aged under five years, per 100 live births',"OBS_VALUE:Observation Value"],key="55")
        
        countries = df["Country"].unique().tolist()
        selected = st.multiselect("Select countries:", countries, default=["Pakistan", "India", "China", "Afghanistan"])
        result = (
        df.groupby(["Year", "Country"])[choice]
        .mean()
        .reset_index()
    )
        plt.figure(figsize=(12,4))
        country=result.where(result['Country'].isin(selected))
        sns.lineplot(data=country,x='Year',y=choice,hue='Country')
        plt.title(f"{choice} over time by Country")
        st.pyplot(plt)

        

    def plot_energy(df):
        countries = df["Country"].unique().tolist()
        selected = st.multiselect("Select countries:", countries, default=["Pakistan", "India", "China", "Afghanistan"])

        # plt.figure(figsize=(6,4))
        for c in selected:
            temp = df[df["Country"] == c].sort_values("Year")
            plt.plot(temp["Year"], temp["Renewable energy consumption"], marker="o", label=c)
        # plt.figure(figsize=(20,20))
        plt.legend()
        plt.grid(True, linestyle="--", alpha=0.6)
        plt.xlabel("Year")
        plt.ylabel("Renewable Energy Consumption by Country")
        plt.title("Renewable Energy Consumption in Selected Countries")
        st.pyplot(plt)

    def heat_map(df):

        numeric_df = df.select_dtypes(include=['number']).drop(columns=['Year'], errors='ignore')
        corr = numeric_df.corr()  


        plt.figure(figsize=(20
                            ,10))
        sns.heatmap(corr, annot=True, cmap="coolwarm", center=0)
        plt.title("Correlation Heatmap")
        st.pyplot(plt)

    


    def Yearly_Trend(df):
        

        choice = st.selectbox("Choose value: ", [
        'Life expectancy at birth, total (years)',
        'Literacy rate, adult total (% of people ages 15 and above)',
        'Renewable energy consumption', 'GDP per capita (current US$)',
        'Annual CO₂ emissions (per capita)',
        'Child mortality rate of children aged under five years, per 100 live births',"OBS_VALUE:Observation Value"],key="44")

        

        yearly_trend = df.groupby("Year")[choice].mean().reset_index()
        plt.figure(figsize=(8,5))
        sns.lineplot(data=yearly_trend, x="Year", y=choice, marker="o")
        plt.title(f"{choice} Over Years")
        plt.ylabel(f"{choice}")
        st.pyplot(plt)

    def top_5(df):
        
        
        choice = st.selectbox("Choose value: ", [
        'Life expectancy at birth, total (years)',
        'Literacy rate, adult total (% of people ages 15 and above)',
        'Renewable energy consumption', 'GDP per capita (current US$)',
        'Annual CO₂ emissions (per capita)',
        'Child mortality rate of children aged under five years, per 100 live births',"OBS_VALUE:Observation Value"],key="22")
        
        choose=st.selectbox("Choose Order: ",["Ascending","Descending"],key='33')

        if choose=="Ascending":
            result=df.groupby('Country')[choice].mean().reset_index()
            Five = result.sort_values(choice, ascending=False).head(5)
            b="Top"
        else:
            result=df.groupby('Country')[choice].mean().reset_index()
            Five = result.sort_values(choice, ascending=True).head(5)
            b="Bottom"
        plt.figure(figsize=(12,6))
        sns.barplot(
            data=Five, 
            x=choice, 
            y="Country", 
            color="skyblue"   
        )

        plt.title(f"{b} 5 Countries by {choice}")
        plt.xlabel(choice)
        plt.ylabel("Country")
        st.pyplot(plt)
        
    def top_vaccine(df):
        plt.figure(figsize=(12,6))
        result=df.groupby(['VACCINE:Vaccine'])['OBS_VALUE:Observation Value'].mean().reset_index()
    
        
        result.plot(x='VACCINE:Vaccine', kind='bar')
        plt.xlabel('Vaccine Type')
        plt.ylabel('Percentage Used')
        plt.title('Vaccine with Most Coverage')
        st.pyplot(plt)
        
    def avg_country(df):
         
        

        choice = st.selectbox("Choose value: ", [
        'Life expectancy at birth, total (years)',
        'Literacy rate, adult total (% of people ages 15 and above)',
        'Renewable energy consumption', 'GDP per capita (current US$)',
        'Annual CO₂ emissions (per capita)',
        'Child mortality rate of children aged under five years, per 100 live births',"OBS_VALUE:Observation Value"],key="country_selectbox")
        
        
        y = st.selectbox("Year: ", [1990, 1991, 1992, 1993, 1994, 1995, 1996, 1997, 1998, 1999, 2000,
            2001, 2002, 2003, 2004, 2005, 2006, 2007, 2008, 2009, 2010, 2011,
            2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022,"All"],key="year_selectbox")
        
        if y=='All':
            result=df.groupby("Country")[choice].mean()
            choose=st.selectbox("Choose Order: ",["Ascending","Descending","Average"])
            # plt.figure(figsize=(12,6))
            b="average"
            if choose=="Ascending":
                result = result.sort_values(ascending=False)
                b= "descending order"
                
            elif choose == "Descending":
                result = result.sort_values(ascending=True)
                b= "ascending order"
            
            elif choose=="Average":
                b="average"
            plt.figure(figsize=(35,6))
            result.plot(kind="bar")   # bar chart
            plt.ylabel(f"{choice} {b}")
            plt.title(f"{choice} {b}")
            st.pyplot(plt)
        else:
            result=df.groupby(['Year','Country'])[choice].mean().reset_index()

            data=result[result["Year"] == y]
            choose=st.selectbox("Choose Order: ",["Ascending","Descending","Average"])
            # plt.figure(figsize=(12,6))
            b="average"
            if choose=="Ascending":
                data = data.sort_values(by=choice,ascending=False)
                b= "descending order"
                
            elif choose == "Descending":
                data = data.sort_values(by=choice,ascending=True)
                b= "ascending order"
            
            elif choose=="Average":
                b="average"
            plt.figure(figsize=(35,6))    
      
            fig, ax = plt.subplots(figsize=(36,4))
            ax.bar(data["Country"], data[choice])

            ax.set_title(f"{choice} Year {y} by Countries ({b})", fontsize=20)
            ax.set_ylabel(f"{choice} {b} for year {y}", fontsize=15)
            ax.set_xlabel("Country", fontsize=15)

            plt.xticks(rotation=90, fontsize=10)  # rotate names

            st.pyplot(fig)
            
    def country_distribution(df):

        choice = st.selectbox("Choose value: ", [
        'Life expectancy at birth, total (years)',
        'Literacy rate, adult total (% of people ages 15 and above)',
        'Renewable energy consumption', 'GDP per capita (current US$)',
        'Annual CO₂ emissions (per capita)',
        'Child mortality rate of children aged under five years, per 100 live births',"OBS_VALUE:Observation Value"],key="11")

        result=df.groupby(['Year','Country'])[choice].mean().reset_index()

        
     
        countries = df["Country"].unique().tolist()
        selected = st.multiselect("Select countries:", countries, default=["Pakistan", "India", "China", "Afghanistan"])
        
        fig, ax = plt.subplots(figsize=(40,16))

        sns.boxplot(
            data=result[result['Country'].isin(selected)],
            x="Country",
            y=choice,
            ax=ax
        )
        ax.set_title(f"Distribution of {choice} acorss years 1990-2022") 
        st.pyplot(fig)


    def metric_comparison(df):
        
        # fig,ax=plt.subplots()
        countries = df["Country"].unique().tolist()
        countries.append("All")
        selected = st.selectbox("Select countries:", countries)

        y = st.selectbox("Year: ", [1990, 1991, 1992, 1993, 1994, 1995, 1996, 1997, 1998, 1999, 2000,
            2001, 2002, 2003, 2004, 2005, 2006, 2007, 2008, 2009, 2010, 2011,
            2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022,"All"])
        if y=="All" and selected != "All":

            df_country=df[df['Country']==selected]

            p=sns.pairplot(df_country[["GDP per capita (current US$)", "Life expectancy at birth, total (years)", "Annual CO₂ emissions (per capita)",'Literacy rate, adult total (% of people ages 15 and above)',
            'Renewable energy consumption','Child mortality rate of children aged under five years, per 100 live births',"OBS_VALUE:Observation Value"]])
            st.pyplot(p.fig)
        elif y=="All" and selected=="All":
            p=sns.pairplot(df[["GDP per capita (current US$)", "Life expectancy at birth, total (years)", "Annual CO₂ emissions (per capita)",'Literacy rate, adult total (% of people ages 15 and above)',
            'Renewable energy consumption','Child mortality rate of children aged under five years, per 100 live births',"OBS_VALUE:Observation Value"]])
            st.pyplot(p.fig)

        elif selected=="All":
            df_year=df[df['Year']==y]
            p=sns.pairplot(df_year[["GDP per capita (current US$)", "Life expectancy at birth, total (years)", "Annual CO₂ emissions (per capita)",'Literacy rate, adult total (% of people ages 15 and above)',
            'Renewable energy consumption','Child mortality rate of children aged under five years, per 100 live births',"OBS_VALUE:Observation Value"]])
            st.pyplot(p.fig)
        elif selected!="All" and y!='All':

            df_year_country = df[(df['Country'] == selected) & (df['Year'] == y)]


            p=sns.pairplot(df_year_country[["GDP per capita (current US$)", "Life expectancy at birth, total (years)", "Annual CO₂ emissions (per capita)",'Literacy rate, adult total (% of people ages 15 and above)',
            'Renewable energy consumption','Child mortality rate of children aged under five years, per 100 live births',"OBS_VALUE:Observation Value"]])
            st.pyplot(p.fig)
                

    def kmean_clusters(df):

        
        


        sel1 = st.selectbox("Between: ", ["2 Metrics","All Metrics"],index=None,placeholder="Select",key="hmm2")

        if sel1:
            if sel1=='2 Metrics':

                sel2 = st.selectbox("Kind: ", ["Country-wise","Yearly","Overall"],index=None,placeholder="Select",key="hm3")
                if sel2:
                    if sel2=="Country-wise":

                        choice1 = st.selectbox("Choose value: ", [
                            'Life expectancy at birth, total (years)',
                            'Literacy rate, adult total (% of people ages 15 and above)',
                            'Renewable energy consumption', 'GDP per capita (current US$)',
                            'Annual CO₂ emissions (per capita)',
                            'Child mortality rate of children aged under five years, per 100 live births',"OBS_VALUE:Observation Value"],index=None,placeholder="Select",key="km1")
                        
                        choice2 = st.selectbox("Choose value: ", [
                            'Life expectancy at birth, total (years)',
                            'Literacy rate, adult total (% of people ages 15 and above)',
                            'Renewable energy consumption', 'GDP per capita (current US$)',
                            'Annual CO₂ emissions (per capita)',
                            'Child mortality rate of children aged under five years, per 100 live births',"OBS_VALUE:Observation Value"],index=None,placeholder="Select",key="km2")

                        countries = df["Country"].unique().tolist()
                        # countries.append("All")

                        if choice1 and choice2:

                            if choice1 !=choice2:
                                selected = st.selectbox("Select country:", countries,index=None,placeholder="Select",key="hmmm2")
                                if selected:
                                    country2=df.groupby(['Year','Country'])[[choice1,choice2
                                    ]].mean().reset_index()
                                    
                                    country2=country2[country2["Country"] == selected]
                                    X = country2[[choice1,choice2]].copy()

                                    # Scale the data (VERY IMPORTANT)
                                    scaler = StandardScaler()
                                    X_scaled = scaler.fit_transform(X)

                                    # Perform K-means clustering
                                    kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
                                    kmeans.fit(X_scaled)

                                    # Add cluster labels back to original dataframe for analysis
                                    country2['cluster'] = kmeans.labels_

                                    # Create the scatter plot
                                    # plt.figure(figsize=(12, 8))
                                    scatter = plt.scatter(country2[choice1], 
                                                        country2[choice2], 
                                                        c=country2['cluster'], 
                                                        cmap='viridis', 
                                                        alpha=0.7,
                                                        s=100)  # s controls point size
                                    centers = scaler.inverse_transform(kmeans.cluster_centers_)  # unscale to original units
                                    plt.scatter(centers[:, 0], centers[:, 1], 
                                                c='red', marker='X', s=300, edgecolor='black', label='Centroids')
                                    plt.xlabel(f'{choice1}', fontsize=12)
                                    plt.ylabel(f'{choice2}', fontsize=12)
                                    plt.title(f'{selected} Clustering: {choice1} vs {choice2}', fontsize=10)
                                    plt.colorbar(scatter, label='Cluster')
                                    plt.grid(True, alpha=0.3)
                                    st.pyplot(plt)
                            else:
                                st.text("Both metric should not be the same")



                    elif sel2=="Yearly":

                        choice1 = st.selectbox("Choose value: ", [
                            'Life expectancy at birth, total (years)',
                            'Literacy rate, adult total (% of people ages 15 and above)',
                            'Renewable energy consumption', 'GDP per capita (current US$)',
                            'Annual CO₂ emissions (per capita)',
                            'Child mortality rate of children aged under five years, per 100 live births',"OBS_VALUE:Observation Value"],index=None,placeholder="Select",key="km1")
                        
                        choice2 = st.selectbox("Choose value: ", [
                            'Life expectancy at birth, total (years)',
                            'Literacy rate, adult total (% of people ages 15 and above)',
                            'Renewable energy consumption', 'GDP per capita (current US$)',
                            'Annual CO₂ emissions (per capita)',
                            'Child mortality rate of children aged under five years, per 100 live births',"OBS_VALUE:Observation Value"],index=None,placeholder="Select",key="km2")

                    
                        
                        if choice1 and choice2:

                            if choice1 !=choice2:
                                y = st.selectbox("Year: ", [1990, 1991, 1992, 1993, 1994, 1995, 1996, 1997, 1998, 1999, 2000,
            2001, 2002, 2003, 2004, 2005, 2006, 2007, 2008, 2009, 2010, 2011,
            2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022],index=None,placeholder="Select",key="hmmm3")
                                if y:
                                    country2=df.groupby(['Year','Country'])[[choice1,choice2
                                    ]].mean().reset_index()
                                    
                                    country2=country2[country2["Year"] == y]
                                    X = country2[[choice1,choice2]].copy()

                                    # Scale the data (VERY IMPORTANT)
                                    scaler = StandardScaler()
                                    X_scaled = scaler.fit_transform(X)

                                    # Perform K-means clustering
                                    kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
                                    kmeans.fit(X_scaled)

                                    # Add cluster labels back to original dataframe for analysis
                                    country2['cluster'] = kmeans.labels_

                                    # Create the scatter plot
                                    # plt.figure(figsize=(12, 8))
                                    scatter = plt.scatter(country2[choice1], 
                                                        country2[choice2], 
                                                        c=country2['cluster'], 
                                                        cmap='viridis', 
                                                        alpha=0.7,
                                                        s=100)  # s controls point size
                                    centers = scaler.inverse_transform(kmeans.cluster_centers_)  # unscale to original units
                                    plt.scatter(centers[:, 0], centers[:, 1], 
                                                c='red', marker='X', s=300, edgecolor='black', label='Centroids')
                                    plt.xlabel(f'{choice1}', fontsize=12)
                                    plt.ylabel(f'{choice2}', fontsize=12)
                                    plt.title(f'Countries Clustering Yearly: {choice1} vs {choice2}', fontsize=10)
                                    plt.colorbar(scatter, label='Cluster')
                                    plt.grid(True, alpha=0.3)
                                    st.pyplot(plt)
                            else:
                                st.text("Both metric should not be the same")


                    elif sel2=="Overall":

                        choice1 = st.selectbox("Choose value: ", [
                            'Life expectancy at birth, total (years)',
                            'Literacy rate, adult total (% of people ages 15 and above)',
                            'Renewable energy consumption', 'GDP per capita (current US$)',
                            'Annual CO₂ emissions (per capita)',
                            'Child mortality rate of children aged under five years, per 100 live births',"OBS_VALUE:Observation Value"],index=None,placeholder="Select",key="km1")
                        
                        choice2 = st.selectbox("Choose value: ", [
                            'Life expectancy at birth, total (years)',
                            'Literacy rate, adult total (% of people ages 15 and above)',
                            'Renewable energy consumption', 'GDP per capita (current US$)',
                            'Annual CO₂ emissions (per capita)',
                            'Child mortality rate of children aged under five years, per 100 live births',"OBS_VALUE:Observation Value"],index=None,placeholder="Select",key="km2")
                        
                        if choice1 and choice2:
                            if choice1!=choice2:

                                country2=df.groupby(['Year','Country'])[[choice1,choice2
             ]].mean().reset_index()
                                X = country2[[choice1, 
                                            choice2]].copy()

                                # Scale the data (VERY IMPORTANT)
                                scaler = StandardScaler()
                                X_scaled = scaler.fit_transform(X)

                                # Perform K-means clustering
                                kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
                                kmeans.fit(X_scaled)

                                # Add cluster labels back to original dataframe for analysis
                                country2['cluster'] = kmeans.labels_

                                # Create the scatter plot
                                # plt.figure(figsize=(12, 8))
                                scatter = plt.scatter(country2[choice1], 
                                                    country2[choice2], 
                                                    c=country2['cluster'], 
                                                    cmap='viridis', 
                                                    alpha=0.7,
                                                    s=100)  # s controls point size
                                centers = scaler.inverse_transform(kmeans.cluster_centers_)  # unscale to original units
                                plt.scatter(centers[:, 0], centers[:, 1], 
                                            c='red', marker='X', s=300, edgecolor='black', label='Centroids')
                                plt.xlabel(f'{choice1}', fontsize=12)
                                plt.ylabel(f'{choice2}', fontsize=12)
                                plt.title(f'Country Clustering: {choice1} vs {choice2}', fontsize=10)
                                plt.colorbar(scatter, label='Cluster')
                                plt.grid(True, alpha=0.3)
                                st.pyplot(plt)

                                # Optional: Print cluster statistics
                                # print("Cluster sizes:")
                                # print(country['cluster'].value_counts().sort_index())
                            else:
                                st.text("Both metric should not be the same")

            else:

                sel2 = st.selectbox("Between: ", ["Country-wise","Yearly","Overall"],index=None,placeholder="Select",key="hmm4")
                if sel2:

                    if sel2=="Overall":

                        # select only numerical columns
                        country2=df.groupby(['Year','Country'])[[ 'Life expectancy at birth, total (years)',
                            'Literacy rate, adult total (% of people ages 15 and above)',
                            'Renewable energy consumption', 'GDP per capita (current US$)',
                            'Annual CO₂ emissions (per capita)',
                            'Child mortality rate of children aged under five years, per 100 live births',"OBS_VALUE:Observation Value"
                            ]].mean().reset_index()
                        
                        X = country2[['GDP per capita (current US$)','Renewable energy consumption','Life expectancy at birth, total (years)','OBS_VALUE:Observation Value',
                'Annual CO₂ emissions (per capita)','Literacy rate, adult total (% of people ages 15 and above)','Child mortality rate of children aged under five years, per 100 live births']]
                        
                        scaler = StandardScaler()
                        X_scaled = scaler.fit_transform(X)

                        kmeans = KMeans(n_clusters=4, random_state=42,n_init=20)  
                        country2['Cluster'] = kmeans.fit_predict(X_scaled)
                        
                        dim=st.selectbox("Number of Dimensions: ",["2","3"])

                        if dim:

                            if dim == "2":
                                pca = PCA(n_components=2)
                                X_pca = pca.fit_transform(X_scaled)

                                plt.scatter(X_pca[:,0], X_pca[:,1], c=country2['Cluster'], cmap='viridis')
                                plt.xlabel('PC1')
                                plt.ylabel('PC2')
                                plt.title('Clusters of Countries Metrics')
                                st.pyplot(plt)
                            else:
                                pca = PCA(n_components=3)
                                X_pca = pca.fit_transform(X_scaled)

                                # Add PCA results to dataframe
                                df_pca = country2.copy()
                                df_pca['PC1'] = X_pca[:, 0]
                                df_pca['PC2'] = X_pca[:, 1]
                                df_pca['PC3'] = X_pca[:, 2]

                                # Plot 3D scatter with Plotly
                                fig = px.scatter_3d(
                                    df_pca,
                                    x='PC1', y='PC2', z='PC3',
                                    color='Cluster',   # Cluster column from k-means
                                    hover_name='Country',  # So when you hover, it shows country name
                                    color_continuous_scale='Viridis'
                                )

                                fig.update_traces(marker=dict(size=6, line=dict(width=0)))
                                fig.update_layout(title='3D PCA Clusters of Countries Metrics')
                                st.plotly_chart(fig, use_container_width=True)
                                
                    elif sel2== "Yearly":

                         # select only numerical columns
                        y = st.selectbox("Year: ", [1990, 1991, 1992, 1993, 1994, 1995, 1996, 1997, 1998, 1999, 2000,
            2001, 2002, 2003, 2004, 2005, 2006, 2007, 2008, 2009, 2010, 2011,
            2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022],index=None,placeholder="Select",key="hmmm3")
                        if y:
                            country2=df.groupby(['Year','Country'])[[ 'Life expectancy at birth, total (years)',
                                'Literacy rate, adult total (% of people ages 15 and above)',
                                'Renewable energy consumption', 'GDP per capita (current US$)',
                                'Annual CO₂ emissions (per capita)',
                                'Child mortality rate of children aged under five years, per 100 live births',"OBS_VALUE:Observation Value"
                                ]].mean().reset_index()
                            country2=country2[country2["Year"]==y]
                            X = country2[['GDP per capita (current US$)','Renewable energy consumption','Life expectancy at birth, total (years)','OBS_VALUE:Observation Value',
                    'Annual CO₂ emissions (per capita)','Literacy rate, adult total (% of people ages 15 and above)','Child mortality rate of children aged under five years, per 100 live births']]
                            
                            scaler = StandardScaler()
                            X_scaled = scaler.fit_transform(X)

                            kmeans = KMeans(n_clusters=4, random_state=42,n_init=20)  
                            country2['Cluster'] = kmeans.fit_predict(X_scaled)
                            
                            dim=st.selectbox("Number of Dimensions: ",["2","3"])

                            if dim:

                                if dim == "2":
                                    pca = PCA(n_components=2)
                                    X_pca = pca.fit_transform(X_scaled)

                                    plt.scatter(X_pca[:,0], X_pca[:,1], c=country2['Cluster'], cmap='viridis')
                                    plt.xlabel('PC1')
                                    plt.ylabel('PC2')
                                    plt.title(f'Clusters of year {y} Metrics')
                                    st.pyplot(plt)
                                else:
                                    pca = PCA(n_components=3)
                                    X_pca = pca.fit_transform(X_scaled)

                                    # Add PCA results to dataframe
                                    df_pca = country2.copy()
                                    df_pca['PC1'] = X_pca[:, 0]
                                    df_pca['PC2'] = X_pca[:, 1]
                                    df_pca['PC3'] = X_pca[:, 2]

                                    # Plot 3D scatter with Plotly
                                    fig = px.scatter_3d(
                                        df_pca,
                                        x='PC1', y='PC2', z='PC3',
                                        color='Cluster',   # Cluster column from k-means
                                        hover_name='Country',  # So when you hover, it shows country name
                                        color_continuous_scale='Viridis'
                                    )

                                    fig.update_traces(marker=dict(size=6, line=dict(width=0)))
                                    fig.update_layout(title=f'3D PCA Clusters of Countries Metrics Year {y}')
                                    st.plotly_chart(fig, use_container_width=True)

                    elif sel2== "Country-wise":

                        
                        countries = df["Country"].unique().tolist()
                        selected = st.selectbox("Select country:", countries,index=None,placeholder="Select",key="hmmm2")

                        if selected:
                            country2=df.groupby(['Year','Country'])[[ 'Life expectancy at birth, total (years)',
                                'Literacy rate, adult total (% of people ages 15 and above)',
                                'Renewable energy consumption', 'GDP per capita (current US$)',
                                'Annual CO₂ emissions (per capita)',
                                'Child mortality rate of children aged under five years, per 100 live births',"OBS_VALUE:Observation Value"
                                ]].mean().reset_index()
                            country2=country2[country2["Country"]==selected]
                            X = country2[['GDP per capita (current US$)','Renewable energy consumption','Life expectancy at birth, total (years)','OBS_VALUE:Observation Value',
                    'Annual CO₂ emissions (per capita)','Literacy rate, adult total (% of people ages 15 and above)','Child mortality rate of children aged under five years, per 100 live births']]
                            
                            scaler = StandardScaler()
                            X_scaled = scaler.fit_transform(X)

                            kmeans = KMeans(n_clusters=4, random_state=42,n_init=20)  
                            country2['Cluster'] = kmeans.fit_predict(X_scaled)
                            
                            dim=st.selectbox("Number of Dimensions: ",["2","3"])

                            if dim:

                                if dim == "2":
                                    pca = PCA(n_components=2)
                                    X_pca = pca.fit_transform(X_scaled)

                                    plt.scatter(X_pca[:,0], X_pca[:,1], c=country2['Cluster'], cmap='viridis')
                                    plt.xlabel('PC1')
                                    plt.ylabel('PC2')
                                    plt.title(f'Clusters of {selected} Metrics')
                                    st.pyplot(plt)
                                else:
                                    pca = PCA(n_components=3)
                                    X_pca = pca.fit_transform(X_scaled)

                                    # Add PCA results to dataframe
                                    df_pca = country2.copy()
                                    df_pca['PC1'] = X_pca[:, 0]
                                    df_pca['PC2'] = X_pca[:, 1]
                                    df_pca['PC3'] = X_pca[:, 2]

                                    # Plot 3D scatter with Plotly
                                    fig = px.scatter_3d(
                                        df_pca,
                                        x='PC1', y='PC2', z='PC3',
                                        color='Cluster',   # Cluster column from k-means
                                        hover_name='Country',  # So when you hover, it shows country name
                                        color_continuous_scale='Viridis'
                                    )

                                    fig.update_traces(marker=dict(size=6, line=dict(width=0)))
                                    fig.update_layout(title=f'3D PCA Clusters of {selected}')
                                    st.plotly_chart(fig, use_container_width=True)


    if choice == "Heatmap":

        st.header("Heatmap")

        heat_map(df)    



    elif choice == "Yearly Country-wise Trends":

        st.header("Yearly Country-wise Trends")

        plot_yearly_country_trend(df)
        

    elif choice =="Country-wise Top and Bottom Trends (Top 5)":

        st.header("Country-wise Top and Bottom Trends (Top 5)")
        top_5(df)

    elif choice=="Vaccine with Most Coverage":
        st.header("Vaccine with Most Coverage")

        top_vaccine(df)     
    elif choice=="Yearly Global Trends":
        st.header("Yearly Global Trends")
        
        Yearly_Trend(df)

    elif choice=="Country-wise Average Trends (All)":
         st.header("Countr-ywise Average Trends (All)")
         
         avg_country(df)
    elif choice == "Country-wise Metrics Distributions":

        st.header("Country-wise Metrics Distributions")
        country_distribution(df)     
    
    elif choice == "Yearly Metric Comparison":
        st.header("Yearly Metric Comparison")
        metric_comparison(df)
    elif choice == "Clusters":
        kmean_clusters(df)    

with tab3:
    st.title("Country Data")



    countries = df["Country"].unique().tolist()
    se=st.selectbox("Select Country ",countries,index=None,placeholder="Select")
    t=m[m['Country']==se]
    t=t.groupby('Year').mean(numeric_only=True)
    if se:
        st.table(t)

        col = st.columns((4, 5), gap='medium')
        with col[0]:
            st.header("Averge stats 1990-2022")
            c_l_m=t['Literacy rate, adult total (% of people ages 15 and above)'].mean()
            st.metric(f"Average Literacy Rate of {se} 1990-2022 %", f"{c_l_m:.1f}%")

            c_l_m=t['Renewable energy consumption'].mean()
            st.metric(f"Average Renewable energy consumption of {se} 1990-2022 ", f"{c_l_m:.1f}")

            c_l_m=t['GDP per capita (current US$)'].mean()
            st.metric(f"Average GDP per capita (current US$) of {se} 1990-2022 ", f"{c_l_m:.0f}$")

            c_l_m=t['Annual CO₂ emissions (per capita)'].mean()
            st.metric(f"Average Annual CO₂ emissions (per capita) of {se} 1990-2022 ", f"{c_l_m:.1f} tons")

            c_l_m=t['Child mortality rate of children aged under five years, per 100 live births'].mean()
            st.metric(f"Average child mortality rate of children aged under five years, per 100 live births of {se} 1990-2022 ", f"{c_l_m:.1f} per 100")

            c_l_m=t['Life expectancy at birth, total (years)'].mean()
            st.metric(f"Average Life expectancy at birth, total (years) of {se} 1990-2022 ", f"{c_l_m:.0f} years")

        with col[1]:
            
            st.header("Year wise stats")
            Years = df["Year"].unique().tolist()
            y=st.selectbox("Select Year ",Years,index=None,placeholder="Select")

            if y:

                y_l=t['Literacy rate, adult total (% of people ages 15 and above)']
                st.metric(f"Literacy Rate of {se} year {y} %", f"{y_l[y]:.1f}%")

                y_l=t['Renewable energy consumption']
                st.metric(f"Renewable energy consumption of {se} year {y} ", f"{y_l[y]:.1f}%")

                y_l=t['GDP per capita (current US$)']
                st.metric(f"GDP per capita (current US$) of {se} year {y} ", f"{y_l[y]:.0f}$")

                y_l=t['Annual CO₂ emissions (per capita)']
                st.metric(f"Annual CO₂ emissions (per capita) of {se} year {y} ", f"{y_l[y]:.1f} tons")

                y_l=t['Child mortality rate of children aged under five years, per 100 live births']
                st.metric(f"Child mortality rate of children aged under five years, per 100 live births of {se} year {y} ", f"{y_l[y]:.1f} per 100")

                y_l=t['Life expectancy at birth, total (years)']
                st.metric(f"Life expectancy at birth, total (years) of {se} year {y} ", f"{y_l[y]:.0f} years")

   

with tab4:

    st.title("Comparison")

    col = st.columns((5, 5, 5), gap='medium')
    prompt=[]
    
    with col[0]:

        countries = df["Country"].unique().tolist()
        se=st.selectbox("Select Country 1 ",countries,index=None,placeholder="Select",key='c1')
        t=m[m['Country']==se]
        t=t.groupby('Year').mean(numeric_only=True)
        if se:
            prompt.append(f"Country 1 {se}")
            st.header("Averge stats 1990-2022")
            c_l_m=t['Literacy rate, adult total (% of people ages 15 and above)'].mean()
            st.metric(f"Average Literacy Rate of {se} 1990-2022 ", f"{c_l_m:.1f}%")
            prompt.append(f"Average Literacy Rate of {se} 1990-2022 : {c_l_m:.1f}%")
            c_l_m=t['Renewable energy consumption'].mean()
            st.metric(f"Average Renewable energy consumption of {se} 1990-2022 ", f"{c_l_m:.1f}%")
            prompt.append(f"Average Renewable energy consumption of {se} 1990-2022 : {c_l_m:.1f}%")
            c_l_m=t['GDP per capita (current US$)'].mean()
            st.metric(f"Average GDP per capita (current US$) of {se} 1990-2022 ", f"{c_l_m:.0f}$")
            prompt.append(f"Average GDP per capita (current US$) of {se} 1990-2022 : {c_l_m:.0f}$")
            
            c_l_m=t['Annual CO₂ emissions (per capita)'].mean()
            st.metric(f"Average Annual CO₂ emissions (per capita) of {se} 1990-2022 ", f"{c_l_m:.1f} tons")
            prompt.append(f"Average Annual CO₂ emissions (per capita) of {se} 1990-2022 : {c_l_m:.1f} tons")
            
            c_l_m=t['Child mortality rate of children aged under five years, per 100 live births'].mean()
            st.metric(f"Average child mortality rate of children aged under five years, per 100 live births of {se} 1990-2022 ", f"{c_l_m:.1f} per 100")
            prompt.append(f"Average child mortality rate of children aged under five years, per 100 live births of {se} 1990-2022 : {c_l_m:.1f} per 100")
            
            c_l_m=t['Life expectancy at birth, total (years)'].mean()
            st.metric(f"Average Life expectancy at birth, total (years) of {se} 1990-2022 ", f"{c_l_m:.0f} years")
            prompt.append(f"Average Life expectancy at birth, total (years) of {se} 1990-2022 : {c_l_m:.0f} years")
            
        


            
            st.header("Year wise stats")
            Years = df["Year"].unique().tolist()
            y=st.selectbox("Select Year ",Years,index=None,placeholder="Select",key='y1')

            if y:

                y_l=t['Literacy rate, adult total (% of people ages 15 and above)']
                st.metric(f"Literacy Rate of {se} year {y} ", f"{y_l[y]:.1f}%")

                y_l=t['Renewable energy consumption']
                st.metric(f"Renewable energy consumption of {se} year {y} ", f"{y_l[y]:.1f}%")

                y_l=t['GDP per capita (current US$)']
                st.metric(f"GDP per capita (current US$) of {se} year {y} ", f"{y_l[y]:.0f}$")

                y_l=t['Annual CO₂ emissions (per capita)']
                st.metric(f"Annual CO₂ emissions (per capita) of {se} year {y} ", f"{y_l[y]:.1f} tons")

                y_l=t['Child mortality rate of children aged under five years, per 100 live births']
                st.metric(f"Child mortality rate of children aged under five years, per 100 live births of {se} year {y} ", f"{y_l[y]:.1f} per 100")

                y_l=t['Life expectancy at birth, total (years)']
                st.metric(f"Life expectancy at birth, total (years) of {se} year {y} ", f"{y_l[y]:.0f} years")


                choice = st.selectbox("Choose value: ", [
                'Life expectancy at birth, total (years)',
                'Literacy rate, adult total (% of people ages 15 and above)',
                'Renewable energy consumption', 'GDP per capita (current US$)',
                'Annual CO₂ emissions (per capita)',
                'Child mortality rate of children aged under five years, per 100 live births',"OBS_VALUE:Observation Value"],key="ch1")
                
            
                result = (
                df.groupby(["Year", "Country"])[choice]
                .mean()
                .reset_index()
            )
                plt.figure(figsize=(12,4))
                country=result.where(result['Country']==se)
                sns.lineplot(data=country,x='Year',y=choice,hue='Country')
                plt.title(f"{choice} over time by Country {se}")
                st.pyplot(plt)

                choice = st.selectbox("Choose value: ", [
                'Life expectancy at birth, total (years)',
                'Literacy rate, adult total (% of people ages 15 and above)',
                'Renewable energy consumption', 'GDP per capita (current US$)',
                'Annual CO₂ emissions (per capita)',
                'Child mortality rate of children aged under five years, per 100 live births',"OBS_VALUE:Observation Value"],key="chb1")

                result=df.groupby(['Year','Country'])[choice].mean().reset_index()

                
            
            
                
                fig, ax = plt.subplots()

                sns.boxplot(
                    data=result[result['Country']==se],
                    x="Country",
                    y=choice,
                    ax=ax
                )
                ax.set_title(f"Distribution of {choice} acorss years 1990-2022 {se}") 
                st.pyplot(fig)

    with col[1]:

        countries = df["Country"].unique().tolist()
        se=st.selectbox("Select Country 2 ",countries,index=None,placeholder="Select",key='c2')
        t=m[m['Country']==se]
        t=t.groupby('Year').mean(numeric_only=True)

        if se:
            prompt.append(f"Country 2 {se}")
            st.header("Averge stats 1990-2022")
            c_l_m=t['Literacy rate, adult total (% of people ages 15 and above)'].mean()
            st.metric(f"Average Literacy Rate of {se} 1990-2022 ", f"{c_l_m:.1f}%")
            prompt.append(f"Average Literacy Rate of {se} 1990-2022 : {c_l_m:.1f}%")
            
            c_l_m=t['Renewable energy consumption'].mean()
            st.metric(f"Average Renewable energy consumption of {se} 1990-2022 ", f"{c_l_m:.1f}%")
            prompt.append(f"Average Renewable energy consumption of {se} 1990-2022 : {c_l_m:.1f}%")


            c_l_m=t['GDP per capita (current US$)'].mean()
            st.metric(f"Average GDP per capita (current US$) of {se} 1990-2022 ", f"{c_l_m:.0f}$")
            prompt.append(f"Average GDP per capita (current US$) of {se} 1990-2022 : {c_l_m:.0f}$")


            c_l_m=t['Annual CO₂ emissions (per capita)'].mean()
            st.metric(f"Average Annual CO₂ emissions (per capita) of {se} 1990-2022 ", f"{c_l_m:.1f} tons")
            prompt.append(f"Average Annual CO₂ emissions (per capita) of {se} 1990-2022 : {c_l_m:.1f} tons")
            
            c_l_m=t['Child mortality rate of children aged under five years, per 100 live births'].mean()
            st.metric(f"Average child mortality rate of children aged under five years, per 100 live births of {se} 1990-2022 ", f"{c_l_m:.1f} per 100")
            prompt.append(f"Average child mortality rate of children aged under five years, per 100 live births of {se} 1990-2022 : {c_l_m:.1f} per 100")
            
            c_l_m=t['Life expectancy at birth, total (years)'].mean()
            st.metric(f"Average Life expectancy at birth, total (years) of {se} 1990-2022 ", f"{c_l_m:.0f} years")
            prompt.append(f"Average Life expectancy at birth, total (years) of {se} 1990-2022 : {c_l_m:.0f} years")


            st.header("Year wise stats")
            Years = df["Year"].unique().tolist()
            y=st.selectbox("Select Year ",Years,index=None,placeholder="Select",key='y2')
           
            if y:
                

                y_l=t['Literacy rate, adult total (% of people ages 15 and above)']
                st.metric(f"Literacy Rate of {se} year {y} ", f"{y_l[y]:.1f}%")

                y_l=t['Renewable energy consumption']
                st.metric(f"Renewable energy consumption of {se} year {y} ", f"{y_l[y]:.1f}%")

                y_l=t['GDP per capita (current US$)']
                st.metric(f"GDP per capita (current US$) of {se} year {y} ", f"{y_l[y]:.0f}$")

                y_l=t['Annual CO₂ emissions (per capita)']
                st.metric(f"Annual CO₂ emissions (per capita) of {se} year {y} ", f"{y_l[y]:.1f} tons")

                y_l=t['Child mortality rate of children aged under five years, per 100 live births']
                st.metric(f"Child mortality rate of children aged under five years, per 100 live births of {se} year {y} ", f"{y_l[y]:.1f} per 100")

                y_l=t['Life expectancy at birth, total (years)']
                st.metric(f"Life expectancy at birth, total (years) of {se} year {y} ", f"{y_l[y]:.0f} years")

                choice = st.selectbox("Choose value: ", [
                'Life expectancy at birth, total (years)',
                'Literacy rate, adult total (% of people ages 15 and above)',
                'Renewable energy consumption', 'GDP per capita (current US$)',
                'Annual CO₂ emissions (per capita)',
                'Child mortality rate of children aged under five years, per 100 live births',"OBS_VALUE:Observation Value"],key="ch2")
                
            
                result = (
                df.groupby(["Year", "Country"])[choice]
                .mean()
                .reset_index()
            )
                plt.figure(figsize=(12,4))
                country=result.where(result['Country']==se)
                sns.lineplot(data=country,x='Year',y=choice,hue='Country')
                plt.title(f"{choice} over time by Country {se}")
                st.pyplot(plt)


                choice = st.selectbox("Choose value: ", [
                'Life expectancy at birth, total (years)',
                'Literacy rate, adult total (% of people ages 15 and above)',
                'Renewable energy consumption', 'GDP per capita (current US$)',
                'Annual CO₂ emissions (per capita)',
                'Child mortality rate of children aged under five years, per 100 live births',"OBS_VALUE:Observation Value"],key="chb2")

                result=df.groupby(['Year','Country'])[choice].mean().reset_index()

                
            
            
                
                fig, ax = plt.subplots()

                sns.boxplot(
                    data=result[result['Country']==se],
                    x="Country",
                    y=choice,
                    ax=ax
                )
                ax.set_title(f"Distribution of {choice} acorss years 1990-2022 {se}") 
                st.pyplot(fig)

    with col[2]:
        
        
        countries = df["Country"].unique().tolist()
        
        se=st.selectbox("Select Country 3 ",countries,index=None,placeholder="Select",key='c')
        t=m[m['Country']==se]
        t=t.groupby('Year').mean(numeric_only=True)
        
        if se:
            prompt.append(f"Country 3 {se}")
            st.header("Averge stats 1990-2022")
            c_l_m=t['Literacy rate, adult total (% of people ages 15 and above)'].mean()
            st.metric(f"Average Literacy Rate of {se} 1990-2022 ", f"{c_l_m:.1f}%")
            prompt.append(f"Average Literacy Rate of {se} 1990-2022 : {c_l_m:.1f}%")
            
            c_l_m=t['Renewable energy consumption'].mean()
            st.metric(f"Average Renewable energy consumption of {se} 1990-2022 ", f"{c_l_m:.1f}%")
            prompt.append(f"Average Renewable energy consumption of {se} 1990-2022 : {c_l_m:.1f}%")


            c_l_m=t['GDP per capita (current US$)'].mean()
            st.metric(f"Average GDP per capita (current US$) of {se} 1990-2022 ", f"{c_l_m:.0f}$")
            prompt.append(f"Average GDP per capita (current US$) of {se} 1990-2022 : {c_l_m:.0f}$")


            c_l_m=t['Annual CO₂ emissions (per capita)'].mean()
            st.metric(f"Average Annual CO₂ emissions (per capita) of {se} 1990-2022 ", f"{c_l_m:.1f} tons")
            prompt.append(f"Average Annual CO₂ emissions (per capita) of {se} 1990-2022 : {c_l_m:.1f} tons")
            
            c_l_m=t['Child mortality rate of children aged under five years, per 100 live births'].mean()
            st.metric(f"Average child mortality rate of children aged under five years, per 100 live births of {se} 1990-2022 ", f"{c_l_m:.1f} per 100")
            prompt.append(f"Average child mortality rate of children aged under five years, per 100 live births of {se} 1990-2022 : {c_l_m:.1f} per 100")
            
            c_l_m=t['Life expectancy at birth, total (years)'].mean()
            st.metric(f"Average Life expectancy at birth, total (years) of {se} 1990-2022 ", f"{c_l_m:.0f} years")
            prompt.append(f"Average Life expectancy at birth, total (years) of {se} 1990-2022 : {c_l_m:.0f} years")

            st.header("Year wise stats")
            Years = df["Year"].unique().tolist()
            y=st.selectbox("Select Year ",Years,index=None,placeholder="Select",key='Y')
           
            if y:
                

                y_l=t['Literacy rate, adult total (% of people ages 15 and above)']
                st.metric(f"Literacy Rate of {se} year {y} ", f"{y_l[y]:.1f}%")

                y_l=t['Renewable energy consumption']
                st.metric(f"Renewable energy consumption of {se} year {y} ", f"{y_l[y]:.1f}%")

                y_l=t['GDP per capita (current US$)']
                st.metric(f"GDP per capita (current US$) of {se} year {y} ", f"{y_l[y]:.0f}$")

                y_l=t['Annual CO₂ emissions (per capita)']
                st.metric(f"Annual CO₂ emissions (per capita) of {se} year {y} ", f"{y_l[y]:.1f} tons")

                y_l=t['Child mortality rate of children aged under five years, per 100 live births']
                st.metric(f"Child mortality rate of children aged under five years, per 100 live births of {se} year {y} ", f"{y_l[y]:.1f} per 100")

                y_l=t['Life expectancy at birth, total (years)']
                st.metric(f"Life expectancy at birth, total (years) of {se} year {y} ", f"{y_l[y]:.0f} years")

                choice = st.selectbox("Choose value: ", [
                'Life expectancy at birth, total (years)',
                'Literacy rate, adult total (% of people ages 15 and above)',
                'Renewable energy consumption', 'GDP per capita (current US$)',
                'Annual CO₂ emissions (per capita)',
                'Child mortality rate of children aged under five years, per 100 live births',"OBS_VALUE:Observation Value"],key="65")
                
            
                result = (
                df.groupby(["Year", "Country"])[choice]
                .mean()
                .reset_index()
            )
                plt.figure(figsize=(12,4))
                country=result.where(result['Country']==se)
                sns.lineplot(data=country,x='Year',y=choice,hue='Country')
                plt.title(f"{choice} over time by Country {se}")
                st.pyplot(plt)

                choice = st.selectbox("Choose value: ", [
                'Life expectancy at birth, total (years)',
                'Literacy rate, adult total (% of people ages 15 and above)',
                'Renewable energy consumption', 'GDP per capita (current US$)',
                'Annual CO₂ emissions (per capita)',
                'Child mortality rate of children aged under five years, per 100 live births',"OBS_VALUE:Observation Value"],key="chb3")

                result=df.groupby(['Year','Country'])[choice].mean().reset_index()

                
            
            
                
                fig, ax = plt.subplots()

                sns.boxplot(
                    data=result[result['Country']==se],
                    x="Country",
                    y=choice,
                    ax=ax
                )
                ax.set_title(f"Distribution of {choice} acorss years 1990-2022 {se}") 
                st.pyplot(fig)
        
with tab5:
    st.title("Future Forecast")

    choose = st.selectbox("Choose : ", ["Global Average Forecast","Country-wise Forecast"],index=None,placeholder="Select")


    def predict_global(df):
    
        # periods = st.number_input("Enter Number of Future Years to predict:", min_value=1, max_value=100, step=1)
        # st.write("Your age is:", age)

        choice = st.selectbox("Choose value: ", [
            'Life expectancy at birth, total (years)',
            'Literacy rate, adult total (% of people ages 15 and above)',
            'Renewable energy consumption', 'GDP per capita (current US$)',
            'Annual CO₂ emissions (per capita)',
            'Child mortality rate of children aged under five years, per 100 live births',"OBS_VALUE:Observation Value"],index=None,placeholder="Select",key="fu1")
        
        if choice:

            periods = st.number_input("Number of Future Years to predict:", min_value=1, max_value=100, step=1)

            df_yearly = df.groupby("Year")[choice].mean().reset_index()

            data = df_yearly.rename(columns={"Year": "ds", choice: "y"})
            data["ds"] = pd.to_datetime(data["ds"], format="%Y")


            model = Prophet()
            model.fit(data)

            
            future = model.make_future_dataframe(periods=periods, freq="YE")
            forecast = model.predict(future)


            y_true = data["y"].values
            y_pred = forecast.loc[: len(data)-1, "yhat"].values  

            mae = mean_absolute_error(y_true, y_pred)
            rmse = np.sqrt(mean_squared_error(y_true, y_pred))

            future_forecast=forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]].tail(periods)
            future_forecast['ds']=future_forecast['ds'].dt.year.astype(str)
            summary = []
            for index, row in future_forecast.iterrows():
                text = f"📅 Year {row['ds']} → Predicted value: {row['yhat']:.2f} (Range: {row['yhat_lower']:.2f} - {row['yhat_upper']:.2f})"
                summary.append(text)
            Textt_summary = ", ".join(summary)  
            prompt_ans= Textt_summary

            client = OpenAI( base_url=endpoint, api_key=api_key, ) 
            response = client.chat.completions.create( messages=[ { "role": "system", "content":
            """You are a data reporting assistant. 
            Your job is to take forecast numbers from Prophet (or any time-series model) 
            and turn them into a human-friendly analysis. 
            Always write in clear, simple English with insights and trends. 
            Mention growth, decline, seasonal effects, and anomalies if they appear.""", }, 
            { "role": "user", "content": prompt_ans, } ], temperature=1,
            top_p=1, model="openai/gpt-4.1-mini" ) 
            st.markdown(response.choices[0].message.content)

            st.text(Textt_summary)  
            st.metric("MAE: ", mae)
            st.metric("RMSE:", rmse)

            plt.figure(figsize=(10,5))
            plt.title(f"Actual vs Predicted {choice} ")
            plt.plot(data["ds"], data["y"], label=f"Actual {choice}")
            plt.plot(forecast["ds"], forecast["yhat"], label=f"Predicted {choice}", color="red")
            plt.fill_between(forecast["ds"], forecast["yhat_lower"], forecast["yhat_upper"], alpha=0.3, color="orange")
            plt.legend()
            st.pyplot(plt)

    def predict_country(df):

        # periods = st.number_input("Enter Number of Future Years to predict:", min_value=1, max_value=100, step=1)
      
        
        choice = st.selectbox("Choose value: ", [
            'Life expectancy at birth, total (years)',
            'Literacy rate, adult total (% of people ages 15 and above)',
            'Renewable energy consumption', 'GDP per capita (current US$)',
            'Annual CO₂ emissions (per capita)',
            'Child mortality rate of children aged under five years, per 100 live births',"OBS_VALUE:Observation Value"],index=None,placeholder="Select",key="fu1")
        
        if choice:
            df_yearly = (
                df.groupby(["Country", "Year"])[choice]
                .mean()
                .reset_index()
            )

        
            countries = df["Country"].unique().tolist()
            se=st.selectbox("Select Country ",countries,index=None,placeholder="Select",key='fc')
            
            
            if se:

                periods = st.number_input("Number of Future Years to predict:", min_value=1, max_value=100, step=1)

                country = df_yearly[df_yearly["Country"] == se]
                
                data = country.rename(columns={"Year": "ds", choice: "y"})

                # Prophet expects ds to be a datetime
                data["ds"] = pd.to_datetime(data["ds"], format="%Y")

                # Fit model
                model = Prophet()
                model.fit(data)

            
                future = model.make_future_dataframe(periods=periods, freq="YE")
                forecast = model.predict(future)

                y_true = data["y"].values
                y_pred = forecast.loc[: len(data)-1, "yhat"].values  

                mae = mean_absolute_error(y_true, y_pred)
                rmse = np.sqrt(mean_squared_error(y_true, y_pred))

                future_forecast=forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]].tail(periods)
                future_forecast['ds']=future_forecast['ds'].dt.year.astype(str)
                summary = []
                for index, row in future_forecast.iterrows():
                    text = f"📅 Year {row['ds']} → Predicted value for {se}: {row['yhat']:.2f} (Range: {row['yhat_lower']:.2f} - {row['yhat_upper']:.2f})"
                    summary.append(text)
                Textt_summary = ", ".join(summary)  

              
                prompt_ans= Textt_summary

                client = OpenAI( base_url=endpoint, api_key=api_key, ) 
                response = client.chat.completions.create( messages=[ { "role": "system", "content": ""
            """You are a data reporting assistant. 
            Your job is to take forecast numbers from Prophet (or any time-series model) 
            and turn them into a human-friendly analysis. 
            Always write in clear, simple English with insights and trends. 
            Mention growth, decline, seasonal effects, and anomalies if they appear.""", }, 
            { "role": "user", "content": prompt_ans, } ], temperature=1,
                top_p=1, model="openai/gpt-4.1-mini" ) 
                st.markdown(response.choices[0].message.content)


               

                st.text(Textt_summary)  
         

                st.metric("MAE: ", mae)
                st.metric("RMSE:", rmse)

                plt.figure(figsize=(10,5))
                plt.title(f"Actual vs Predicted {choice} for {se} ")
                plt.plot(data["ds"], data["y"], label=f"Actual {choice}")
                plt.plot(forecast["ds"], forecast["yhat"], label=f"Predicted {choice}", color="red")
                plt.fill_between(forecast["ds"], forecast["yhat_lower"], forecast["yhat_upper"], alpha=0.3, color="orange")
                plt.legend()
                st.pyplot(plt)

    if choose =="Global Average Forecast":

        predict_global(df)   

    elif choose=="Country-wise Forecast":

        predict_country(df)    

with tab6:
    prompt.append(".")
    # prompt.append("Again you generated a report despite their was no country mentioned in the query")

    result = ", ".join(prompt)
    st.text(result)
    
    if st.button("📊 AI Insights"):

        if result==".":
            st.header("No Data Feeded")

        else:
            # client = ollama.Client()


            # model = "data_analyst"  
            # prompt_ans= result


            # response = client.generate(model=model, prompt=prompt_ans)
            

            # # st.text_area("AI Response", response.response, height=200, disabled=True)
            # # st.code(response.response, language="markdown")
            # st.markdown(response.response)
           
            client = OpenAI( base_url=endpoint, api_key=api_key, ) 
            response = client.chat.completions.create( messages=[ { "role": "system", "content": ""
            """You are an AI data analyst.
You generate **structured, professional insights** with headings, bullet points, and future predictions.
Always provide clear recommendations for countries with low performance.** Also Provide future recommendations for all countries for their improvement **.
if their are no countries in the query just say ** no data feeded **""", }, 
            { "role": "user", "content": result, } ], temperature=1,
    top_p=1, model=model ) 
            st.markdown(response.choices[0].message.content)

  
