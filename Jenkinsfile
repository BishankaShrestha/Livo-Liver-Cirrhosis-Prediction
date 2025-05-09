pipeline {
    agent any

    environment {
        IMAGE_NAME = "liver-app"
        CONTAINER_PORT = "8501"
    }

    stages {
        stage('Clone Repo') {
            steps {
                git url: 'https://github.com/BishankaShrestha/Livo-Liver-Cirrhosis-Prediction.git'
            }
        }

        stage('Build Docker Image') {
            steps {
                sh 'docker build -t $IMAGE_NAME .'
            }
        }

        stage('Stop Existing Container') {
            steps {
                script {
                    def containerId = sh(script: "docker ps -q --filter ancestor=$IMAGE_NAME", returnStdout: true).trim()
                    if (containerId) {
                        sh "docker stop $containerId"
                        sh "docker rm $containerId"
                    }
                }
            }
        }

        stage('Run Docker Container') {
            steps {
                sh 'docker run -d -p 8501:8501 $IMAGE_NAME'
            }
        }
    }

    post {
        success {
            echo "App deployed successfully!"
        }
        failure {
            echo "Build failed."
        }
    }
}
