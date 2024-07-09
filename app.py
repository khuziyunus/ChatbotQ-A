import streamlit as st
from sentence_transformers import SentenceTransformer, util
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

nltk.download('punkt')
nltk.download('stopwords')

def preprocess(text):
    tokens = word_tokenize(text.lower())
    tokens = [word for word in tokens if word.isalnum()]
    tokens = [word for word in tokens if word not in stopwords.words('english')]
    return ' '.join(tokens)

# Load a pre-trained sentence transformer model
model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

qa_pairs = {
    "Are you facing issue with login?": "Please email to moeez.aslam@fatima-group.com with screenshot or you can try resetting password by clicking on forgot password and typing in your username on next page. You will receive an email with link to set new password.",
    "What is password policy?": "You need to change password every [duration].",
    "Is your portal showing error page?": "Try to login on another computer or phone. Also clear caches of browser.",
    "Do you need to know payment status or escalate payment?": "You can check the status from 'invoices/payments' tab or call your PO buyer. Click on Finance Tab and then select Invoice Status tab. Enter PO number only and click on Go button.",
    "Is your invoice in rejected state?": "You can email your buyer and ask reason for rejection. You can not re-enter invoice in rejected state. You have to get it cancelled for re-entry.",
    "Do you want to cancel invoice?": "Please email to your buyer so that he can get the invoice cancelled from finance.",
    "Have you made a mistake and want to re-enter?": "Please cancel invoice in system by emailing to buyer.",
    "Is your payment getting delayed?": "You can check status from PO buyer. In order for smooth/fast process, you must submit you invoice on portal before dispatching hardcopy to finance. Make sure everything is in order: invoice number is correct, invoice date is correct, attachments are correct, right quantity is selected, tax applied is correct, total with tax on portal matches hardcopy invoice. Invoice process include: portal entry, hardcopy receive by finance, finance starts processing for 3 way matching after first 2 requirements, invoices gets approved, voucher is created, check is created and then check is signed before dispatching to finance. Standard time is 30 days after finance has received correct invoice hardcopy and correct portal upload. If all things are in order payment might be received sooner than 30 days.",
    "Do you have single invoice and multiple MRR?": "You will follow your invoice. You will create single entry and select all lines that are included in invoice hardcopy.",
    "Do you have single invoice and multiple PO?": "Enter PO and click on go and select line. After this click on Add to invoice button. Then enter next PO and click on go and select line. Then click on add invoice. When all lines are added as per hardcopy invoice click on next button to proceed to next page.",
    "Is your invoice quantity different from MRR quantity?": "Select the MRR line that is in your invoice. On the next page where you enter invoice number, there is editable option near available quantity in Lines section. Change quantity to match your hardcopy invoice.",
    "Do you get quantity unavailable error?": "One of the selected lines is already invoiced. Please check invoice status and do not select that line.",
    "Do you get unique invoice number error?": "You might have entered same invoice number before. Invoice numbers must be unique. You may resolve this by adding '#1'/'#2'/'#3' with the invoice number.",
    "Do you want to enter freight on portal?": "First please make sure that Freight/Miscellaneous charges are mentioned on PO. Your company invoice must have freight charges on it along with goods. While entering on portal, on the page where you enter invoice number, go to page bottom and click on plus sign near shipping and handling. From the drop down menu select either freight or miscellaneous depending on what is mentioned on PO and enter amount in amount column. Do not change anything else.",
    "Do you want to add more than 30 lines of invoice on portal at the same time?": "Select all the 30 lines with the check bow on top of table and then click on 'Add to invoice' button. Then scroll down and select the remaining lines and again click on 'Add to invoice' button. You can check number of lines from right side of page. When all lines are added you can proceed with invoicing to the next page.",
    "Are you not able to see PO lines while searching PO?": "Following are the reasons for no results:\n- MRR is not created (Ask you PO buyer to support in creating MRR)\n- PO lines are closed (Ask you PO buyer to open PO lines)\n- Ffert (Unit 3) PO (These POs usually have different series than FFL POs. Please enter 'Fatim' in organization after entering PO number and wait. List of values will appear. Select FATIMAFERT OU and then click on go button)\n- PO belongs to a different supplier code (This happens when multiple supplier entities exist in database)\n- PO is in re-approval state (Contact your PO buyer)",
    "Do you need to change tax code on your invoice?": "On step 3 of your invoice submission, there is Tax Line Summary Table. If there are more than 3 rows, you won't be able to fix tax code. If there are exactly 3 rows, go to the line that states (sales tax/code(st)). Please ignore the lines that says (excise and rec_st). In the row that has sales tax, go to tax code column and click on the magnifying glass near the tax code (it may state st_(18+0) or no tax). When you click on magnifying near the mentioned code a new window will pop up. Completely remove the code from the top bar in new window. Now write (%18% or %25% whichever is applicable) in the bar and click on go button. Select the code (st_(18+0) or st_(25+0) whichever is applicable. You will come back to the table from where you started. Now click on Calculate button on top left corner of this table. After this go to bottom of page and click on Recalculate Total button. Submit the invoice if total on your hardcopy invoice and portal total matches exactly.",
    "Did you invoice got rejected or cancelled?": "You can email to find out reason for rejection/cancellation. Invoice is rejected due to following issues:\n- Invoice number do not match hardcopy invoice (Invoice number must be same as sales tax invoice number)\n- Multiple invoices attached (Invoice must be entered one at a time, no matter how many POs or MRRs)\n- Wrong attachments (Attachments must be correct)\n- Invoice total does not match hardcopy (Total must match before you submit on portal)\n- Invoice date is wrong (Invoice date on portal must be same as invoice date on sales tax invoice)",
    "Are you having trouble locating draft quote?": "Click on full list button on sourcing page near active/draft responses table and find your quote by entering RFQ number in search bar.",
    "Do you want changes in your company database (address, username, phone number, etc)?": "Please contact your PO buyer for required changes.",
    "Do you want to access responsibility?": "Click on setting (Gear Icon) and select access request. Click on the FG procurement responsibility and select the required access. Enter justification and submit for approval."
}

questions = list(qa_pairs.keys())
answers = list(qa_pairs.values())

encoded_questions = model.encode(questions, convert_to_tensor=True)

def advanced_chatbot(user_input):
    processed_input = preprocess(user_input)
    encoded_input = model.encode(processed_input, convert_to_tensor=True)
    similarities = util.pytorch_cos_sim(encoded_input, encoded_questions)
    best_match_idx = similarities.argmax().item()
    if similarities[0, best_match_idx] < 0.5:
        return "I'm sorry, I didn't understand that. Can you please rephrase?"
    return answers[best_match_idx]

st.title("Q/A Chatbot")
st.write("Ask a question related to the QA pairs")

user_input = st.text_input("Your question:")
if user_input:
    response = advanced_chatbot(user_input)
    st.write("Chatbot response:", response)
