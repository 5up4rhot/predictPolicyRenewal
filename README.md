### Structure
- app: dockerized flask application
- ml: research notebooks

### Usage
1. Build and run docker container
```
docker build -t pred_policy_renewal app/
docker run --rm -p 5000:5000 pred_policy_renewal
```
2. Open http://0.0.0.0:5000/