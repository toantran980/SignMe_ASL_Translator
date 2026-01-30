# SignMe ASL Translator

Set up:

npx create-expo-app@latest --template blank ./

Download Expo Go

npx expo start

Install and edit conf follow guide (Install Expo Router ): https://docs.expo.dev/router/installation/#modify-project-configuration


Docker: 

Deep Clean if old conf. 

```
docker compose down --rmi all --volumes --remove-orphansdocker compose up --build
```

Start and Rebuild Fresh

```
docker compose up --build
```

- Note: without --build if no change

Accessing the App:

- Scan the QR code
