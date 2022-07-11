import com.github.jengelman.gradle.plugins.shadow.tasks.ShadowJar

val dl4jVersion:String by project
val nd4jVersion:String by project

plugins {
    kotlin("jvm") version "1.6.10"
    application
    id("com.github.johnrengelman.shadow") version "5.0.0"
}

group = "me.singlerr"
version = "1.0-SNAPSHOT"

repositories {
    mavenCentral()
    maven {
        url = uri("https://jitpack.io")
    }
}


dependencies {
    implementation(kotlin("stdlib"))
    implementation(group = "me.tongfei", name = "progressbar", version = "0.9.3")
    implementation(group = "com.github.shin285", name = "KOMORAN", version = "3.3.4")
    implementation(group = "org.json", name = "json", version = "20211205")
    // implementation(group = "org.nd4j", name = "nd4j-native", version = nd4jVersion)
    // implementation(group = "org.nd4j", name = "nd4j-native", version = nd4jVersion, classifier = "linux-x86_64-avx2")
    //implementation(group = "org.nd4j", name = "nd4j-native-platform", version = nd4jVersion)
    //  implementation(group = "org.nd4j", name = "nd4j-cuda-10.0", version = nd4jVersion)
    //  implementation(group = "org.nd4j", name = "nd4j-cuda-10.0", version = nd4jVersion, classifier="windows-x86_64")
    implementation(group = "org.nd4j", name = "nd4j-cuda-11.2", version = nd4jVersion)
    implementation("org.deeplearning4j:deeplearning4j-modelimport:1.0.0-M2")
    implementation(group = "org.deeplearning4j", name="deeplearning4j-ui", version= dl4jVersion)
    implementation(group = "org.deeplearning4j", name = "deeplearning4j-cuda-11.2", version = dl4jVersion)
    implementation(group = "org.deeplearning4j", name = "deeplearning4j-core", version = dl4jVersion)
    implementation(group = "org.deeplearning4j", name = "deeplearning4j-nlp", version = dl4jVersion)
    implementation(group = "org.junit.jupiter", name = "junit-jupiter", version = "5.7.0")
    implementation(group = "ch.qos.logback", name = "logback-classic", version = "1.2.6")
}
project.setProperty("mainClassName","dev.isteam.chatbot.MainKt")
tasks.withType<ShadowJar>{
    isZip64 = true
    archiveClassifier.set("")
}
group = "dev.isteam"
version = "1.0-SNAPSHOT"
description = rootProject.name
java.sourceCompatibility = JavaVersion.VERSION_11
tasks.build{
    dependsOn("shadowJar")
}
/*
application{
    mainClass.set("dev.isteam.chatbot.Main")
}

 */
