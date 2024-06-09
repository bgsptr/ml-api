running in ngrok:
	https://bea2-182-253-51-211.ngrok-free.app

curl pOSt:
	curl --header "Content-Type: application/json" \
	--request POST \
	--data '{"email":"agunggusti95@gmail.com","password":"satu", "fname":"satu", "lname":"lname"}' \
	http://localhost:5000/register

	curl --header "Content-Type: application/json" \
	--request POST \
	--data '{"email":"agunggusti95@gmail.com","password":"satu"}' \
	https://bea2-182-253-51-211.ngrok-free.app/login


	curl --header "Content-Type: application/json" \
	--request POST \
	--data '{"email":"agunggusti95@gmail.com","otp":"131535"}' \
	http://localhost:5000/email/otp

#POST BASE_URL/api/v1/beneficiaries

# server Basic U0ltTWlkLXNlcnZlci0zNU15andmbzNxaVgzb0hpdkJMUlRiSGY6
# client Basic U0ltTWlkLWNsaWVudC1vY045b0FHMmljeFVhN1pJOg==
	
	curl -v --header "Content-Type: application/json" \
	--header "Accept: application/json" \
	--header "Authorization:Basic TWlkLWNsaWVudC05MHB0Y3NIeTh2SGxPekZDOg==" \
	--request POST \
	--data '{"name":"Putrawan","account":"081338494371","bank":"gopay","alias_name":"wawa","email":"agunggusti95@gmail.com"}' \
	https://app.midtrans.com/iris/api/v1/beneficiaries

	curl -v --header "Content-Type: application/json" \
	--header "Accept: application/json" \
	--header "Authorization:Basic TWlkLWNsaWVudC05MHB0Y3NIeTh2SGxPekZDOg==" \
	--request POST \
	--data '{"beneficiary_name":"Gopay Simulator A","beneficiary_account":"08123450000","bank":"gopay","amount":"wawa","1000.00":"notes:payout june 8"}' \
	https://app.midtrans.com/iris/api/v1/payouts


curl -v https://app.sandbox.midtrans.com/iris/ping

curl https://f8a8-182-253-51-211.ngrok-free.app/google_auth

ngrok config add-authtoken 2hZwZKakmNtLjg6x3vL89ipH1pq_3YEGduUXKrTqS4qXYkqCa