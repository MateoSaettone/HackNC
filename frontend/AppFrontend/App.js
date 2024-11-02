import React, { useEffect, useState } from 'react';
import { Text, View } from 'react-native';
import axios from 'axios';

// To start: yarn start

export default function App() {
  const [message, setMessage] = useState('');

  useEffect(() => {
    axios.get('http://127.0.0.1:8000/')
      .then(response => {
        setMessage(response.data.message);
      })
      .catch(error => {
        console.error('Error connecting to backend:', error);
      });
  }, []);

  return (
    <View style={{ flex: 1, justifyContent: 'center', alignItems: 'center' }}>
      <Text>{message || 'Loading...'}</Text>
    </View>
  );
}
