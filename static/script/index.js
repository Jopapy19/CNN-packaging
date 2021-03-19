// remove this comment
let base64Image;   //binary-to-text encoding schemes 
$("#image-selector").change(function() {
    let reader = new FileReader(); // call objet
    reader.onload = function(e) {
        let dataURL = reader.result;
        $('#selected-image').attr('src', dataURL);
        base64Image = dataURL.replace(
            "data:image/png;base64,","");
    }
    reader.onload.readAsDataURL($("#image-selector")[0].files[0]);
    $("#men-prediction").text("");
    $("#women-prediction").text("");
});

$("#prediction-button").click(function(){
    let message = {
        image: base64Image
    }
    fetch(`${window.origin}/predict`, {
        method: 'POST', // *GET, POST, PUT, DELETE, etc.
        headers : {
            'Content-Type': 'application/json;'
        },
        body: JSON.stringify(message) // Body data type must match "Content-Type" header
    }).then(response => response.json()).then(function(response){
        $("#men-prediction").text(response.prediction.men);
        $("#women-prediction").text(response.prediction.women);
    });
});