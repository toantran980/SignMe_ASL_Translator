# SignMe ASL Translator


Set up EXPO (without docker):

npx create-expo-app@latest --template blank ./

Download Expo Go

npx expo start

Install and edit conf follow guide (Install Expo Router ): [https://docs.expo.dev/router/installation/#modify-project-configuration](https://docs.expo.dev/router/installation/#modify-project-configuration)




**Docker (use this)**

Deep Clean if old conf.

```
docker compose down --rmi all --volumes --remove-orphansdocker compose up --build
```

Close and clean docker instance:
```
docker compose down -v
```

Start and Rebuild Fresh

```
docker compose up --build
```

* Note: remove --build if no change

Accessing the App:

* Scan the QR code
