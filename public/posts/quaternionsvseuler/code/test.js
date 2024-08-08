import * as THREE from 'three';
import { TrackballControls } from 'three/addons/controls/TrackballControls.js';

const container = document.getElementById('canvas1');

const scene = new THREE.Scene();
const camera = new THREE.PerspectiveCamera( 75, container.clientWidth / container.clientHeight, 0.1, 1000 );

const renderer = new THREE.WebGLRenderer();
renderer.setPixelRatio( window.devicePixelRatio );
renderer.setSize( container.clientWidth, container.clientHeight );
renderer.setAnimationLoop( animate );
container.append( renderer.domElement );

const geometry = new THREE.BoxGeometry( 1, 1, 1 );
const material = new THREE.MeshBasicMaterial( { color: 0xffff00 } );
const cube = new THREE.Mesh( geometry, material );
scene.add( cube );

camera.position.z = 5;

const controls = new TrackballControls(camera, renderer.domElement);
controls.update();

function animate() {

	cube.rotation.x -= 0.02;
	cube.rotation.y -= 0.02;

    controls.update();

	renderer.render( scene, camera );

}