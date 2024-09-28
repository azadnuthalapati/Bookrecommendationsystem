from flask import Flask,render_template,request
import pickle 
import numpy as np

popular_df = pickle.load(open('popular_books_186.pkl','rb'))
popular_df_186 = pickle.load(open('popular_books_186.pkl','rb'))
pt = pickle.load(open('pt.pkl','rb'))
books = pickle.load(open('books.pkl','rb'))
similarity_scores = pickle.load(open('similarity_scores.pkl','rb'))
knn_model = pickle.load(open('knn_model.pkl','rb'))

app = Flask(__name__)

def get_random_books():
    return popular_df.sample(30)

@app.route('/')
def home():
    # Randomly select 30 books to display
    random_books = get_random_books()

    return render_template(
        'index.html',
        random_books=random_books,
        book_name=list(random_books['Book-Title'].values),
        author=list(random_books['Book-Author'].values),
        image=list(random_books['Image-URL-M'].values),
        votes=list(random_books['num_ratings'].values),
        rating=list(random_books['avg_ratings'].values)
    )

@app.route('/top/<int:number_of_books>')
def top_books(number_of_books):
    # Sort and select top n books
    top_books = popular_df.head(number_of_books)
    
    return render_template(
        'books.html',
        book_name=list(top_books['Book-Title'].values),
        author=list(top_books['Book-Author'].values),
        image=list(top_books['Image-URL-M'].values),
        votes=list(top_books['num_ratings'].values),
        rating=list(top_books['avg_ratings'].values),
        number_of_books=number_of_books
    )
@app.route('/recommend')
def recommend_ui():
    return render_template('recommend.html')

@app.route('/recommend_books',methods=['post'])
def recommend():
    user_input = request.form.get('user_input')
    button_e1 = request.form.get('engine_1')
    button_e2 = request.form.get('engine_2')
    # Handle empty user input
    if not user_input:
        return render_template('recommend.html', message="Please choose a book name.")
    
    try:
        index = np.where(pt.index == user_input)[0][0]
    except IndexError:
        return render_template('recommend.html', message="Book not found. Please choose a valid book name.")

    if button_e1:
        similar_items = sorted(list(enumerate(similarity_scores[index])), key=lambda x: x[1], reverse=True)[1:5]
        data = []
        for i in similar_items:
            item = []
            temp_df = books[books['Book-Title'] == pt.index[i[0]]]
            item.extend(list(temp_df.drop_duplicates('Book-Title')['Book-Title'].values))
            item.extend(list(temp_df.drop_duplicates('Book-Title')['Book-Author'].values))
            item.extend(list(temp_df.drop_duplicates('Book-Title')['Image-URL-M'].values))

            data.append(item)
        return render_template('recommend.html',data=data)
    elif button_e2:
        distances,suggestions = knn_model.kneighbors(pt.iloc[index,:].values.reshape(1,-1),n_neighbors=6)
        data = []
        for i in suggestions[0][1:]:
            item = []
            temp_df = books[books['Book-Title'] == pt.index[i]]
            item.extend(list(temp_df.drop_duplicates('Book-Title')['Book-Title'].values))
            item.extend(list(temp_df.drop_duplicates('Book-Title')['Book-Author'].values))
            item.extend(list(temp_df.drop_duplicates('Book-Title')['Image-URL-M'].values))
            data.append(item)
        return render_template('recommend.html',data=data)

if __name__ == '__main__':
    app.run(debug=True)
