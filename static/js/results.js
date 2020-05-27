$(function(){
	$('button').click(function(){
      // var img = $('#inputFileName').val();
      // console.log('image', img);
      // var filename = $('#inputFileName').val();
      var username = $('#inputUsername').val();
      console.log('image', username);
		$.ajax({
			url: '/queryImage',
			data: $('form').serialize(),
			type: 'POST',
			success: function(response){
				console.log(response);
				console.log('SUCCESS');
			},
			error: function(error){
				console.log(error);
				console.log('WWWTTTFFF');
			}
		});
	});
});