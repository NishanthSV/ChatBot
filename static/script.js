function submitMessage() {
    var inputText = document.getElementById("inputText").value;
    var conversation = document.getElementById("conversation");
    
    document.getElementById("inputText").value = "";
    
    var userInputDiv = document.createElement("div");
    userInputDiv.classList.add("message", "user");
    userInputDiv.textContent = "\n"+inputText;
    conversation.appendChild(userInputDiv);
    
    conversation.scrollTop = conversation.scrollHeight;
    
    var xhr = new XMLHttpRequest();
    xhr.onreadystatechange = function() {
        if (xhr.readyState == XMLHttpRequest.DONE) {
            if (xhr.status == 200) {
                var response = JSON.parse(xhr.responseText).response;
                
                conversation.scrollTop = conversation.scrollHeight;

                var botResponseDiv = document.createElement("div");
                botResponseDiv.classList.add("message", "response");
                botResponseDiv.textContent = "\n" + response;
                conversation.appendChild(botResponseDiv);
                
                conversation.scrollTop = conversation.scrollHeight;
            } else {
                console.error("ML inference request failed: " + xhr.status);
            }
        }
    };
    xhr.open("POST", "/predict", true);
    xhr.setRequestHeader("Content-Type", "application/x-www-form-urlencoded");
    xhr.send("inputText=" + encodeURIComponent(inputText));
}
