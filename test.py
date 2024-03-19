from flask import redirect

def redirect_to_youtube(url):
    # Vérifiez si l'URL commence par "https://www.youtube.com/" pour garantir qu'elle est valide
    if url.startswith("https://www.youtube.com/"):
        # Redirige vers l'URL YouTube fournie
        return redirect(url)
    else:
        return "L'URL fournie n'est pas une URL YouTube valide."

url = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
redirect_to_youtube(url)
