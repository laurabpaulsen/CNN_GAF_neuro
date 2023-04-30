To download the data it is necessary to obtain a authentication key from openneuro.org. When this is done run the following commands from the terminal:

```
curl -sL https://deb.nodesource.com/setup_14.x | sudo -E bash -
sudo apt-get install -y nodejs
sudo npm install -g @openneuro/cli
```

Then login using your authentication key from openneuro by using the following command and then providing your API key (can be obtained for free by creating an account on openneuro.org)
```
openneuro login
```

Now the data can be downloaded by using the following comand. 
```
openneuro download --snapshot 2.0.0 ds004018 ds004018-download/
```
After the download is finished, make sure the data is in the ´data´ directory, and rename it to `raw` instead of `ds004018-download`