import streamlit.components.v1 as components
import base64

def paste_handler():
    # This component injects Javascript that listens for the global 'paste' event
    # If it detects an image, it sends it back as a Base64 string
    js_code = """
    <script>
    document.addEventListener('paste', function (e) {
        var items = e.clipboardData.items;
        for (var i = 0; i < items.length; i++) {
            if (items[i].type.indexOf('image') !== -1) {
                var blob = items[i].getAsFile();
                var reader = new FileReader();
                reader.onload = function(event) {
                    window.parent.postMessage({
                        type: 'streamlit:setComponentValue',
                        value: event.target.result
                    }, '*');
                };
                reader.readAsDataURL(blob);
            }
        }
    });
    </script>
    <div style="height:0px; width:0px; overflow:hidden;">Paste interceptor active</div>
    """
    return components.html(js_code, height=0, width=0)
