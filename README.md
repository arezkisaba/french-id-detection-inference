# Steps to follow to Authorize HTTPS access from Ubuntu with LAN address instead of 127.0.0.1
mkcert -install  
mkcert 192.168.1.100  
sudo cp ./192.168.1.100.pem ./192.168.1.100.crt  
sudo cp ./192.168.1.100.crt ./192.168.1.100.cer  
sudo cp ./192.168.1.100.crt /usr/local/share/ca-certificates/  
sudo update-ca-certificates  
sudo ls /etc/ssl/certs | grep 192.168.1.100  
sudo chmod 644 /usr/local/share/ca-certificates/192.168.1.100.crt  
Send 192.168.1.100.crt by mail  
Install Profile  
Certificate Trust Settings > Check your certificate  

# Steps to follow to Authorize HTTPS access from iPhone with LAN address instead of 127.0.0.1
mkcert -CAROOT  
$HOME/.local/share/mkcert/rootCA.pem  
openssl x509 -outform der -in $HOME/.local/share/mkcert/rootCA.pem -out ./rootCA.crt  
Send rootCA.crt by mail  
Install Profile  
Certificate Trust Settings > Check your certificate  

# More
edge://settings  
edge://settings/clearBrowserData (Cached images and files)  
edge://net-internals/#dns > clear  
#openssl s_client -connect 192.168.1.100:12345 -CAfile /etc/ssl/certs/192.168.1.100.pem  
