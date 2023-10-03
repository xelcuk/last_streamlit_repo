import numpy as np
import pandas as pd
import streamlit as st


# Sayfa Ayarları
st.set_page_config(
    page_title="Fraud Classifier",
    page_icon="https://unalarif.com/wp-content/uploads/image-181.png",
    menu_items={
        "About": "For More Information\n" + "https://www.linkedin.com/in/selçuk-türk-13b817248/"
    }
)


# Başlık Ekleme
st.title(" **:red[Fraud]** Transaction Project")

# Markdown Oluşturma
st.markdown("This investigation deals with whether a transaction is **:red[fraud]** or **:green[not]**.")

# Resim Ekleme
st.image("https://www.webtekno.com/images/editor/default/0003/79/66289949824b83934c667c258b7a7cc0ee3b65ee.jpeg")

st.markdown("We developed a **:blue[machine learning]** model to help people with their research")
st.markdown("Let's check it !!!")

st.image("https://portal.buluttahsilat.com/static_files/docs/11/notificationimage//71/7fff3b2b-35a4-453d-9ed8-5c28cc53d71d.png")


st.markdown(" ")
st.markdown("Some places in **Europe** where our model has been used and validated by people")
df2 = pd.DataFrame(
    np.random.randn(7000, 2) / [1,0.26] + [50.76, 16.4],
    columns=['lat', 'lon'])

st.map(df2)


# Header Ekleme
st.header("Data Dictionary")

st.markdown("- **type** : Type of online transaction (1 = PAYMENT, 2 = CASH_OUT, 3 = CASH_IN, 4 = TRANSFER, 5 = DEBIT)")
st.markdown("- **amount** : The amount of the transaction")
st.markdown("- **oldbalanceOrg** : Balance before the transaction")
st.markdown("- **newbalanceOrig** : Balance after the transaction")

# Pandasla veri setini okuyalım
df = pd.read_csv("frauddf.csv")



# Küçük bir düzenleme :)
df.amount = df.amount.astype(int)
df.oldbalanceOrg= df.oldbalanceOrg.astype(int)
df.newbalanceOrig= df.newbalanceOrig.astype(int)




# Sidebarda Markdown Oluşturma
st.sidebar.markdown("**Choose** the features below to see the result!")

# Sidebarda Kullanıcıdan Girdileri Alma
name = st.sidebar.text_input("Name", help="Please capitalize the first letter of your name!")
surname = st.sidebar.text_input("Surname", help="Please capitalize the first letter of your surname!")
age= st.sidebar.slider("Age", min_value=0, max_value=130)
gender = st.sidebar.selectbox(
    'Gender',
    ('Female','Male'))
st.write('You selected:', gender)

banks = st.sidebar.multiselect(
    'Which banks do you prefer for your transactions?',
    ['BMO', 'NTC', 'KLS', 'TYR','KWW','IST'])


amount= st.sidebar.number_input("The amount of the transaction", min_value=0, format="%d")
oldbalanceOrg = st.sidebar.number_input("Balance before the transaction", min_value=0)
newbalanceOrig= oldbalanceOrg-amount
type = st.sidebar.number_input("Type of online transaction", min_value=1, max_value=5)


# Pickle kütüphanesi kullanarak eğitilen modelin tekrardan kullanılması
from joblib import load
xgb_model = load('xgboost_model.pkl')


input_df = pd.DataFrame({
    'Type': [type],
    'Amount': [amount],
    'OldbalanceOrg': [oldbalanceOrg],
    'NewbalanceOrig': [newbalanceOrig]
})

pred = xgb_model.predict(input_df.values)
pred_probability = np.round(xgb_model.predict_proba(input_df.values), 3)



st.header("Results")

# Sonuç Ekranı
if st.sidebar.button("Submit"):

    # Info mesajı oluşturma
    st.info("You can find the result below.")

    # Sorgulama zamanına ilişkin bilgileri elde etme
    from datetime import date, datetime

    today = date.today()
    # dd/mm/YY
    today = today.strftime("%d/%m/%Y")

    time = datetime.now().strftime("%H:%M:%S")

    # Sonuçları Görüntülemek için DataFrame
    results_df = pd.DataFrame({
    'Name': [name],
    'Surname': [surname],
    'Date': [today],
    'Time': [time],
    'Type': [type],
    'Amount': [amount],
    'OldbalanceOrg': [oldbalanceOrg],
    'NewbalanceOrig': [newbalanceOrig],
    'Prediction': [pred],
    #'NoFraud Probability': [pred_probability[:,:1]],
    #'Fraud Probability': [pred_probability[:,1:]]
    })

    results_df["Prediction"] = results_df["Prediction"].apply(lambda x: str(x).replace("0","NoFraud"))
    results_df["Prediction"] = results_df["Prediction"].apply(lambda x: str(x).replace("1","Fraud"))



    st.table(results_df)

    if pred == 0:
        st.image("https://cdn.imgbin.com/16/24/10/imgbin-partnership-open-business-partner-computer-icons-business-mahZe5BTWQH9Dcm7gkwBVdnqt.jpg")

        audio_file = open('money-soundfx.mp3', 'rb')
        audio_bytes = audio_file.read()

        st.audio(audio_bytes, format='money-soundfx.mp3')
        
    else:
        st.image("https://media.istockphoto.com/id/1269117710/photo/fraud-alert-conceptual-warning-road-sign-against-stormy-sky.jpg?b=1&s=612x612&w=0&k=20&c=UjjX8MWaQSFWHabwYBCjqJmIE4PL0X-iCH-nNMGEvlM=")
        
        
        audio_file = open('fbi.mp3', 'rb')
        audio_bytes = audio_file.read()

        st.audio(audio_bytes, format='fbi.mp3')


else:
    st.markdown("Please click the *Submit Button*!")