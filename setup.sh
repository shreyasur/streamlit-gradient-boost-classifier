mkdir -p ~/.streamlit/
echo "\
[server]\n\
headless = true\n\
port = $PORT\n\
enableCORS = false\n\
[theme]\n\
base = "dark"\n\
primaryColor = "#fffa86"\n\
\n\
" > ~/.streamlit/config.toml
