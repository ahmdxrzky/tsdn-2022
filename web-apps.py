# 1. Status
# 2. Total_Good_Debt & Bad Debt
# 2. Job_Title
# 3. Applicant_Age
# 4. Applicant_Gender
# 5. Housing_Type
# 6. Total_Income

# Import libraries
import streamlit as st
import pandas as pd
import numpy as np
import requests
import io
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_selection import mutual_info_regression
from sklearn.preprocessing import LabelEncoder
from scipy import stats

# Heatmap
def heatmap_visualization(data, title, pad, cmap="Blues", 
                          t_fontsize=30, x_fontsize=15,
                          weight="bold"):
    """Heatmap Visualization."""  
    # Figure, Axes, Subplots
    fig, ax = plt.subplots(1, 1,
                           figsize=(20, 12),
                           facecolor="#FAF7F0")
    # Heatmap
    sns.heatmap(data.corr(),
                annot=True,
                cmap=cmap,
                ax=ax)
    # Set title
    ax.set_title(title,
                 weight=weight,
                 fontsize=t_fontsize,
                 pad=pad)
    # Ticklabels
    plt.xticks(weight=weight,
               fontsize=x_fontsize)  
    plt.yticks(weight=weight,
               fontsize=x_fontsize)
    ax.set_facecolor("#FAF7F0")
    
    return fig

# Get Mutual Information function
def get_plot_mi_score(X_mutual, y_mutual, discrete_features):
    mi_scores = mutual_info_regression(X_mutual, y_mutual, discrete_features=discrete_features)
    mi_scores = pd.Series(mi_scores, name="MI Scores", index=X_mutual.columns)
    mi_scores = mi_scores.sort_values(ascending=False)
    scores = mi_scores.sort_values(ascending=True)
    width = np.arange(len(scores))
    ticks = list(scores.index)
    fig, ax = plt.subplots(1, 1, figsize=(20, 12), facecolor="#FAF7F0")
    plt.barh(width, scores, color="g", edgecolor="k")
    plt.yticks(width, ticks, weight="bold", fontsize=10)
    plt.xticks(weight="bold", fontsize=10)
    ax.set_facecolor("#FAF7F0")
    plt.title("Mutual Information Scores",
            weight="bold",
            fontsize=20,
            pad=25)
    return fig

# Plotting bivariate analysis between target and a numerical feature
def bivariate_numerical_plot(data, x, hue, title, xlabel, ylabel):
    fig, ax = plt.subplots(1, 1,
                           figsize=(20, 6), 
                           facecolor="#FAF7F0")
    sns.kdeplot(data=data, 
                x=x, 
                hue=hue, 
                fill=True)
    
    ax.set_facecolor("#FAF7F0")
    
    plt.title(title, weight="bold", 
              fontsize=25, pad=30)
    plt.xticks(weight="bold", fontsize=10)
    plt.yticks(weight="bold", fontsize=10)
    plt.xlabel(xlabel, weight="bold", 
               fontsize=15, labelpad=15)
    plt.ylabel(ylabel, weight="bold", 
               fontsize=15, labelpad=15)
    
    return fig

# 
def numerical_plotting(data, col, title, symb, ylabel, color):
    fig, ax = plt.subplots(2, 1, 
                           sharex=True, 
                           figsize=(20, 8),
                           facecolor="#FAF7F0",
                           gridspec_kw={"height_ratios": (.2, .8)})
    
    ax[0].set_facecolor("#FAF7F0")
    ax[1].set_facecolor("#FAF7F0")
    
    ax[0].set_title(title, 
                    weight="bold", 
                    fontsize=30, 
                    pad=30)
    
    sns.boxplot(x=col, 
                data=data,
                color=color,
                ax=ax[0])
    
    ax[0].set(yticks=[])
    
    sns.distplot(data[col], kde=True, color=color)
    
    plt.xticks(weight="bold", fontsize=10)
    plt.yticks(weight="bold", fontsize=10)
    
    ax[0].set_xlabel(col, weight="bold", fontsize=15, labelpad=15)
    ax[1].set_xlabel(col, weight="bold", fontsize=15)
    ax[1].set_ylabel(ylabel, weight="bold", fontsize=15)
    
    plt.axvline(data[col].mean(), 
                color='darkgreen', 
                linewidth=2.2, 
                label='mean=' + str(np.round(data[col].mean(),1)) + symb)
    plt.axvline(data[col].median(), 
                color='red', 
                linewidth=2.2, 
                label='median='+ str(np.round(data[col].median(),1)) + symb)
    plt.axvline(data[col].max(), 
                color='blue', 
                linewidth=2.2, 
                label='max='+ str(np.round(data[col].max(),1)) + symb)
    plt.axvline(data[col].min(), 
                color='orange', 
                linewidth=2.2, 
                label='min='+ str(np.round(data[col].min(),1)) + symb)
    plt.axvline(data[col].mode()[0], 
                color='purple', 
                linewidth=2.2, 
                label='mode='+ str(data[col].mode()[0]) + symb)
    
    
    plt.legend(bbox_to_anchor=(1, 1), 
               ncol=1, 
               fontsize=17, 
               fancybox=True, 
               shadow=True, 
               frameon=False)
    
    plt.tight_layout()
    
    return fig

st.set_page_config(page_title="YGGDRASIL Web Apps", layout="wide", initial_sidebar_state="auto")
st.markdown(
    f"""
    <style>
    .appview-container .main .block-container{{
        padding-top: 2rem;
        padding-right: 3rem;
        padding-left: 3rem;
        padding-bottom: 2rem;}}
    </style>
    """,
    unsafe_allow_html=True
)

def main():
    st.markdown("<h1 style='text-align: center;'> YGGDRASIL Web Apps </h1>", unsafe_allow_html=True)

    # Load dataset
    url = "https://raw.githubusercontent.com/ahmdxrzky/data-tsdn-2022/main/application_data.csv"
    download = requests.get(url).content
    application = pd.read_csv(io.StringIO(download.decode('utf-8')))
    application_copy = application.copy()

    st.markdown("<h4 style='text-align: center;'> Dataset \"Credit Card Approval\" <h4>", unsafe_allow_html=True)
    st.markdown("""
        <p style='text-align: center;'>
        Dataset ini berasal dari Kaggle dan dapat diakses di <a href="https://www.kaggle.com/datasets/rikdifos/credit-card-approval-prediction">sini</a>.
        </p>
    """, unsafe_allow_html=True)
    st.dataframe(application_copy)

    st.markdown("""
        <p style='text-align: justify;'>
        Dataset di atas memiliki 21 kolom.
        Kolom <i><b>status</b></i> menyatakan status approval kartu kredit dari applicant tersebut, sehingga fitur ini menjadi fitur target.
        Agar fitur-fitur kategorik dapat dianalisis dengan baik, maka dilakukanlah encoding dengan <i><b>Label Encoder</b></i>.
        </p>
    """, unsafe_allow_html=True)
    # Encode Data with Label Encoder
    label_encoder = LabelEncoder() # label encoder
    for i in application_copy.columns:
        if application_copy[i].dtype == "object":
            label_encoder.fit_transform(list(application_copy[i].values))
            application_copy[i] = label_encoder.transform(application_copy[i].values)
    del application_copy["Applicant_ID"]

    st.markdown("<h4 style='text-align: center;'> Korelasi <h4>", unsafe_allow_html=True)
    st.markdown("""
        <p style='text-align: justify;'>
        Untuk mempermudah, analisis korelasi perlu dilakukan untuk menentukan fitur mana yang paling berpengaruh terhadap nilai dari fitur status.
        </p>
    """, unsafe_allow_html=True)

    X_mutual = application_copy.astype("int64").reset_index().copy()
    y_mutual = X_mutual.pop("Status")
    del X_mutual["index"]
    discrete_features = X_mutual.dtypes == int

    fig_heatmap = heatmap_visualization(application_copy.corr(), "Application Correlation", pad=30);
    st.pyplot(fig_heatmap)
    st.markdown("""
        <p style='text-align: justify;'>
        Dengan menggunakan heatmap di atas, fitur <b><i>Total_Bad_Debt</b></i> terlihat memiliki korelasi negatif yang kuat dengan kolom status.
        </p>
    """, unsafe_allow_html=True)

    fig_mi_score = get_plot_mi_score(X_mutual, y_mutual, discrete_features=discrete_features)
    st.pyplot(fig_mi_score)
    st.markdown("""
        <p style='text-align: justify;'>
        Hasil serupa juga didapatkan pada analisis mutual information
        Mutual Information menunjukkan nilai ketergantungan antara 2 variabel.
        </p>
    """, unsafe_allow_html=True)

    fig_bad_debt = numerical_plotting(data=application, 
                                      col="Total_Bad_Debt", 
                                      title="Applicant Total Bad Debt Distribution",
                                      symb=' ', 
                                      ylabel="Density", 
                                      color="#AABF0F");
    st.pyplot(fig_bad_debt)

    # Correlation:
    corr_pear, _ = stats.pearsonr(application["Total_Bad_Debt"], application["Status"])
    corr_spear, p_spear = stats.spearmanr(application["Total_Bad_Debt"], application["Status"])

    st.write(f"""
        Fitur 'Total_Bad_Debt' dengan 'Status' memiliki Korelasi Pearson dan Spearman berturut-turut {corr_pear:.3f} dan {corr_spear:.3f}.\n
        Apabila H0 menyatakan bahwa 2 sampel tidak berkorelasi,""")
    alpha = 0.05
    if p_spear > alpha:
        st.write(f"maka terima H0 karena p = {p_spear:.3f}")
    else:
        st.write(f"maka tolak H0 karena p = {p_spear:.3f}")

    # calculate kendall's correlation
    coef_kendall, p_kendall = stats.kendalltau(application["Total_Bad_Debt"], application["Status"]) 
    st.write(f"""
        Sementara itu, Korelasi Kendall dari kedua fitur sebesar {coef_kendall:.3f},""")
    if p_kendall > alpha:
        st.write(f"sehingga terima H0 karena p = {p_spear:.3f}")
    else:
        st.write(f"sehingga tolak H0 karena p = {p_spear:.3f}")

    fig_bivariate = bivariate_numerical_plot(data=application, 
                                             x="Total_Bad_Debt",
                                             hue="Status",
                                             title="Applicant Total Bad Debt",
                                             xlabel="Applicant Total Bad Debt",
                                             ylabel="Density");
    st.pyplot(fig_bivariate)

if __name__ == "__main__":
    main()