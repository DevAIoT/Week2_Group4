import * as THREE from 'three';

export class GolfSimulator {
    constructor(container) {
        this.container = container;
        this.scene = null;
        this.camera = null;
        this.renderer = null;
        this.controls = null;
        
        // Golf course elements
        this.ground = null;
        this.tee = null;
        this.hole = null;
        this.flag = null;
        
        // Trajectory visualization
        this.trajectoryLine = null;
        this.trajectoryPoints = [];
        this.ball = null;
        this.ballTrail = [];
        
        // Animation
        this.animationId = null;
        this.clock = new THREE.Clock();
        this.isPaused = false;
        
        // Camera settings
        this.cameraTarget = new THREE.Vector3(0, 0, 0);
        this.defaultCameraPosition = new THREE.Vector3(50, 30, 50);
    }

    async init() {
        this.createScene();
        this.createCamera();
        this.createRenderer();
        this.createLights();
        this.createGolfCourse();
        this.createTrajectorySystem();
        this.setupControls();
        this.startAnimation();
        
        console.log('Golf Simulator initialized');
    }

    createScene() {
        this.scene = new THREE.Scene();
        this.scene.background = new THREE.Color(0x87CEEB); // Sky blue
        
        // Add fog for depth perception
        this.scene.fog = new THREE.Fog(0x87CEEB, 100, 500);
    }

    createCamera() {
        const aspect = this.container.clientWidth / this.container.clientHeight;
        this.camera = new THREE.PerspectiveCamera(75, aspect, 0.1, 1000);
        this.camera.position.copy(this.defaultCameraPosition);
        this.camera.lookAt(this.cameraTarget);
    }

    createRenderer() {
        this.renderer = new THREE.WebGLRenderer({ 
            antialias: true,
            alpha: true 
        });
        this.renderer.setSize(this.container.clientWidth, this.container.clientHeight);
        this.renderer.setPixelRatio(window.devicePixelRatio);
        this.renderer.shadowMap.enabled = true;
        this.renderer.shadowMap.type = THREE.PCFSoftShadowMap;
        this.renderer.outputColorSpace = THREE.SRGBColorSpace;
        
        this.container.appendChild(this.renderer.domElement);
    }

    createLights() {
        // Ambient light
        const ambientLight = new THREE.AmbientLight(0x404040, 0.4);
        this.scene.add(ambientLight);
        
        // Main directional light (sun)
        const directionalLight = new THREE.DirectionalLight(0xffffff, 1);
        directionalLight.position.set(100, 100, 50);
        directionalLight.castShadow = true;
        directionalLight.shadow.mapSize.width = 2048;
        directionalLight.shadow.mapSize.height = 2048;
        directionalLight.shadow.camera.near = 0.5;
        directionalLight.shadow.camera.far = 500;
        directionalLight.shadow.camera.left = -100;
        directionalLight.shadow.camera.right = 100;
        directionalLight.shadow.camera.top = 100;
        directionalLight.shadow.camera.bottom = -100;
        this.scene.add(directionalLight);
        
        // Fill light
        const fillLight = new THREE.DirectionalLight(0xffffff, 0.3);
        fillLight.position.set(-50, 50, -50);
        this.scene.add(fillLight);
    }

    createGolfCourse() {
        // Create ground (fairway)
        const groundGeometry = new THREE.PlaneGeometry(400, 200);
        const groundMaterial = new THREE.MeshLambertMaterial({ 
            color: 0x228B22,
            side: THREE.DoubleSide 
        });
        this.ground = new THREE.Mesh(groundGeometry, groundMaterial);
        this.ground.rotation.x = -Math.PI / 2;
        this.ground.receiveShadow = true;
        this.scene.add(this.ground);
        
        // Create tee area
        const teeGeometry = new THREE.CircleGeometry(2, 16);
        const teeMaterial = new THREE.MeshLambertMaterial({ color: 0x8B4513 });
        this.tee = new THREE.Mesh(teeGeometry, teeMaterial);
        this.tee.rotation.x = -Math.PI / 2;
        this.tee.position.set(0, 0.01, 0);
        this.scene.add(this.tee);
        
        // Create hole and green (distant target)
        const greenGeometry = new THREE.CircleGeometry(8, 32);
        const greenMaterial = new THREE.MeshLambertMaterial({ color: 0x006400 });
        const green = new THREE.Mesh(greenGeometry, greenMaterial);
        green.rotation.x = -Math.PI / 2;
        green.position.set(0, 0.02, 150);
        this.scene.add(green);
        
        // Create hole
        const holeGeometry = new THREE.CircleGeometry(0.2, 16);
        const holeMaterial = new THREE.MeshLambertMaterial({ color: 0x000000 });
        this.hole = new THREE.Mesh(holeGeometry, holeMaterial);
        this.hole.rotation.x = -Math.PI / 2;
        this.hole.position.set(0, 0.03, 150);
        this.scene.add(this.hole);
        
        // Create flag
        const flagPoleGeometry = new THREE.CylinderGeometry(0.05, 0.05, 3, 8);
        const flagPoleMaterial = new THREE.MeshLambertMaterial({ color: 0xFFFFFF });
        const flagPole = new THREE.Mesh(flagPoleGeometry, flagPoleMaterial);
        flagPole.position.set(0, 1.5, 150);
        flagPole.castShadow = true;
        this.scene.add(flagPole);
        
        const flagGeometry = new THREE.PlaneGeometry(1.5, 1);
        const flagMaterial = new THREE.MeshLambertMaterial({ 
            color: 0xFF0000,
            side: THREE.DoubleSide 
        });
        this.flag = new THREE.Mesh(flagGeometry, flagMaterial);
        this.flag.position.set(0.75, 2.5, 150);
        this.scene.add(this.flag);
        
        // Add some decorative elements
        this.createTrees();
        this.createDistanceMarkers();
    }

    createTrees() {
        const treePositions = [
            [-80, 0, 50], [80, 0, 60], [-60, 0, 120], [70, 0, 130],
            [-90, 0, -20], [85, 0, -30], [-50, 0, 180], [60, 0, 170]
        ];
        
        treePositions.forEach(pos => {
            // Tree trunk
            const trunkGeometry = new THREE.CylinderGeometry(0.5, 0.8, 5, 8);
            const trunkMaterial = new THREE.MeshLambertMaterial({ color: 0x8B4513 });
            const trunk = new THREE.Mesh(trunkGeometry, trunkMaterial);
            trunk.position.set(pos[0], 2.5, pos[2]);
            trunk.castShadow = true;
            this.scene.add(trunk);
            
            // Tree foliage
            const foliageGeometry = new THREE.SphereGeometry(3, 8, 6);
            const foliageMaterial = new THREE.MeshLambertMaterial({ color: 0x228B22 });
            const foliage = new THREE.Mesh(foliageGeometry, foliageMaterial);
            foliage.position.set(pos[0], 7, pos[2]);
            foliage.castShadow = true;
            this.scene.add(foliage);
        });
    }

    createDistanceMarkers() {
        const distances = [50, 100, 150];
        
        distances.forEach(distance => {
            const markerGeometry = new THREE.PlaneGeometry(2, 0.5);
            const markerMaterial = new THREE.MeshLambertMaterial({ 
                color: 0xFFFFFF,
                side: THREE.DoubleSide 
            });
            const marker = new THREE.Mesh(markerGeometry, markerMaterial);
            marker.position.set(-15, 1, distance);
            marker.lookAt(-15, 1, 0);
            this.scene.add(marker);
            
            // Add text (simplified)
            const textGeometry = new THREE.PlaneGeometry(1.5, 0.3);
            const textMaterial = new THREE.MeshLambertMaterial({ 
                color: 0x000000,
                side: THREE.DoubleSide 
            });
            const text = new THREE.Mesh(textGeometry, textMaterial);
            text.position.set(-15, 0.5, distance);
            text.lookAt(-15, 0.5, 0);
            this.scene.add(text);
        });
    }

    createTrajectorySystem() {
        // Create ball
        const ballGeometry = new THREE.SphereGeometry(0.2, 16, 16);
        const ballMaterial = new THREE.MeshLambertMaterial({ color: 0xFFFFFF });
        this.ball = new THREE.Mesh(ballGeometry, ballMaterial);
        this.ball.position.set(0, 0.2, 0);
        this.ball.castShadow = true;
        this.ball.visible = false;
        this.scene.add(this.ball);
        
        // Create trajectory line material
        this.trajectoryLineMaterial = new THREE.LineBasicMaterial({
            color: 0xFF4444,
            linewidth: 3,
            transparent: true,
            opacity: 0.8
        });
    }

    setupControls() {
        // Simple mouse controls for camera
        let isMouseDown = false;
        let mouseX = 0;
        let mouseY = 0;
        let cameraAngleX = 0;
        let cameraAngleY = 0;
        
        this.renderer.domElement.addEventListener('mousedown', (event) => {
            if (event.button === 0) {
                isMouseDown = true;
                mouseX = event.clientX;
                mouseY = event.clientY;
            }
        });
        
        this.renderer.domElement.addEventListener('mousemove', (event) => {
            if (!isMouseDown) return;
            
            const deltaX = event.clientX - mouseX;
            const deltaY = event.clientY - mouseY;
            
            cameraAngleY -= deltaX * 0.01;
            cameraAngleX -= deltaY * 0.01;
            cameraAngleX = Math.max(-Math.PI/3, Math.min(Math.PI/3, cameraAngleX));
            
            const radius = 60;
            this.camera.position.x = radius * Math.cos(cameraAngleY) * Math.cos(cameraAngleX);
            this.camera.position.y = radius * Math.sin(cameraAngleX) + 20;
            this.camera.position.z = radius * Math.sin(cameraAngleY) * Math.cos(cameraAngleX);
            
            this.camera.lookAt(this.cameraTarget);
            
            mouseX = event.clientX;
            mouseY = event.clientY;
        });
        
        this.renderer.domElement.addEventListener('mouseup', () => {
            isMouseDown = false;
        });
        
        // Zoom with mouse wheel
        this.renderer.domElement.addEventListener('wheel', (event) => {
            const zoomSpeed = 0.1;
            const direction = this.camera.position.clone().sub(this.cameraTarget).normalize();
            
            if (event.deltaY > 0) {
                this.camera.position.add(direction.multiplyScalar(zoomSpeed * 5));
            } else {
                this.camera.position.sub(direction.multiplyScalar(zoomSpeed * 5));
            }
            
            event.preventDefault();
        });
    }

    displayTrajectory(trajectoryData) {
        this.clearTrajectory();
        
        if (!trajectoryData || !trajectoryData.points || trajectoryData.points.length === 0) {
            return;
        }
        
        // Create trajectory line
        const points = trajectoryData.points.map(point => 
            new THREE.Vector3(point.x, point.y, point.z)
        );
        
        const geometry = new THREE.BufferGeometry().setFromPoints(points);
        this.trajectoryLine = new THREE.Line(geometry, this.trajectoryLineMaterial);
        this.scene.add(this.trajectoryLine);
        
        // Show ball at start position
        if (points.length > 0) {
            this.ball.position.copy(points[0]);
            this.ball.visible = true;
        }
        
        // Animate ball along trajectory
        this.animateBallTrajectory(trajectoryData);
        
        console.log(`Trajectory displayed: ${trajectoryData.points.length} points`);
    }

    animateBallTrajectory(trajectoryData) {
        if (!trajectoryData.points || trajectoryData.points.length === 0) return;
        
        let currentPointIndex = 0;
        const animationSpeed = 1.0; // Adjust for faster/slower animation
        
        const animate = () => {
            if (currentPointIndex >= trajectoryData.points.length) {
                return; // Animation complete
            }
            
            const point = trajectoryData.points[currentPointIndex];
            this.ball.position.set(point.x, point.y, point.z);
            
            // Create trail effect
            if (this.ballTrail.length > 20) {
                const oldTrail = this.ballTrail.shift();
                this.scene.remove(oldTrail);
            }
            
            const trailGeometry = new THREE.SphereGeometry(0.1, 8, 8);
            const trailMaterial = new THREE.MeshBasicMaterial({ 
                color: 0xFFAAAA,
                transparent: true,
                opacity: 0.5 
            });
            const trail = new THREE.Mesh(trailGeometry, trailMaterial);
            trail.position.copy(this.ball.position);
            this.ballTrail.push(trail);
            this.scene.add(trail);
            
            currentPointIndex += Math.ceil(animationSpeed);
            
            setTimeout(animate, 50); // 20 FPS animation
        };
        
        animate();
    }

    clearTrajectory() {
        if (this.trajectoryLine) {
            this.scene.remove(this.trajectoryLine);
            this.trajectoryLine = null;
        }
        
        this.ball.visible = false;
        
        // Clear ball trail
        this.ballTrail.forEach(trail => {
            this.scene.remove(trail);
        });
        this.ballTrail = [];
        
        console.log('Trajectory cleared');
    }

    resetCamera() {
        this.camera.position.copy(this.defaultCameraPosition);
        this.camera.lookAt(this.cameraTarget);
        console.log('Camera reset');
    }

    startAnimation() {
        const animate = () => {
            if (!this.isPaused) {
                this.animationId = requestAnimationFrame(animate);
                
                // Update flag animation
                if (this.flag) {
                    const time = this.clock.getElapsedTime();
                    this.flag.rotation.y = Math.sin(time * 2) * 0.1;
                }
                
                this.renderer.render(this.scene, this.camera);
            }
        };
        
        animate();
    }

    pause() {
        this.isPaused = true;
        if (this.animationId) {
            cancelAnimationFrame(this.animationId);
        }
    }

    resume() {
        this.isPaused = false;
        this.startAnimation();
    }

    handleResize() {
        const width = this.container.clientWidth;
        const height = this.container.clientHeight;
        
        this.camera.aspect = width / height;
        this.camera.updateProjectionMatrix();
        
        this.renderer.setSize(width, height);
        
        console.log(`Renderer resized to ${width}x${height}`);
    }

    // Utility methods
    getScene() {
        return this.scene;
    }

    getCamera() {
        return this.camera;
    }

    getRenderer() {
        return this.renderer;
    }
}