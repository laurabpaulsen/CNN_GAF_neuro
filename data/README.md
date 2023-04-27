To download the data it is necessary to obtain a authentication key from openneuro.org. When this is done run the following commands from the terminal:

```
curl -sL https://deb.nodesource.com/setup_14.x | sudo -E bash -
sudo apt-get install -y nodejs
sudo npm install -g @openneuro/cli
```

Then login using your authentication key from openneuro
```
openneuro login
```

Now the data can be downloaded by using the following comand. 
```
openneuro download --snapshot 1.0.4 ds004504 ds004504-download/ 
```