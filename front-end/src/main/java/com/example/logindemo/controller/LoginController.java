package com.example.logindemo.controller;

import com.example.logindemo.dto.LoginRequest;
import com.example.logindemo.model.DiseaseStatistics;
import com.example.logindemo.model.PatientStatistics;
import com.example.logindemo.service.LoginService;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Controller;
import org.springframework.ui.Model;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestParam;

import java.util.ArrayList;
import java.util.List;

@Controller
public class LoginController {
    @Autowired
    private LoginService loginService;

    @GetMapping("/login")
    public String showLoginForm() {
        return "login";
    }

    @PostMapping("/login")
    public String login(@RequestParam("username") String username,
                        @RequestParam("password") String password,
                        Model model) {
        LoginRequest loginRequest = new LoginRequest();
        loginRequest.setUsername(username);
        loginRequest.setPassword(password);

        if (loginService.validateUser(loginRequest)) {
            return "redirect:/index"; // 登录成功跳转到主页
        } else {
            model.addAttribute("error", "Invalid username or password");
            return "login";
        }
    }
    // 新增方法处理 /index 请求
    @GetMapping("/index")
    public String showIndex() {
        return "index"; // 返回 index.html 页面
    }
    @GetMapping("/register")
    public String showRegister() {
        return "register"; // 返回 register.html 页面
    }
    @GetMapping("/dashboard")
    public String dashboard() {
        return "dashboard"; // 返回 templates/dashboard.html
    }

    @GetMapping("/add-patient")
    public String addPatient() {
        return "add-patient"; // 返回 templates/add-patient.html
    }

    @GetMapping("/patient-list")
    public String patientList() {
        return "patient-list"; // 返回 templates/patient-list.html
    }

    @GetMapping("/data-query")
    public String dataQuery() {
        return "data-query"; // 返回 templates/data-query.html
    }

    @GetMapping("/fundus-image-history")
    public String fundusImageHistory() {
        return "fundus-image-history"; // 返回 templates/fundus-image-history.html
    }

    @GetMapping("/eye-image-upload")
    public String eyeImageUpload() {
        return "eye-image-upload"; // 返回 templates/eye-image-upload.html
    }

    @GetMapping("/add-doctor")
    public String addDoctor() {
        return "add-doctor"; // 返回 templates/add-doctor.html
    }

    @GetMapping("/doctor-list")
    public String doctorList() {
        return "doctor-list"; // 返回 templates/doctor-list.html
    }

    @GetMapping("/statistics")
    public String statistics() {
        return "statistics"; // 返回 templates/statistics.html
    }

    @GetMapping("/change-password")
    public String changePassword() {
        return "change-password"; // 返回 templates/change-password.html
    }

    @GetMapping("/system-help")
    public String systemHelp() {
        return "system-help"; // 返回 templates/system-help.html
    }

    @GetMapping("/personal-info")
    public String personalInfo() {
        return "personal-info"; // 返回 templates/personal-info.html
    }

    @GetMapping("/UploadDualEyeImages")
    public String showUploadDualEyeImages() {
        return "UploadDualEyeImages"; // 返回 templates/UploadDualEyeImages.html
    }
    @GetMapping("/eye-image-upload-batch")
    public String eyeImageUploadBatch() {
        return "eye-image-upload-batch"; // 返回 templates/eye-image-upload-batch.html
    }
    @GetMapping("/eye-image-upload-single")
    public String eyeImageUploadSingle() {
        return "eye-image-upload-single"; // 返回 templates/eye-image-upload-batch.html
    }
    @GetMapping("/single-detail")
    public String singleImageDetail() {
        return "single-detail";
    }

    @GetMapping("/batch-detail")
    public String batchImageDetail() {
        return "batch-detail";
    }
//    @GetMapping("/statistics")
//    public String showStatistics(Model model) {
//        // 设置当前激活菜单
//        model.addAttribute("activeMenu", "statistics");
//
//        // 设置医生名称
//        model.addAttribute("doctorName", "张医生");
//
//        // 模拟患者统计数据
//        List<PatientStatistics> patientStats = new ArrayList<>();
//        patientStats.add(new PatientStatistics("2016", 4292));
//        patientStats.add(new PatientStatistics("2017", 5432));
//        patientStats.add(new PatientStatistics("2018", 6423));
//        patientStats.add(new PatientStatistics("2019", 7253));
//        patientStats.add(new PatientStatistics("2020", 8133));
//        patientStats.add(new PatientStatistics("2021", 7932));
//        patientStats.add(new PatientStatistics("2022", 8232));
//        patientStats.add(new PatientStatistics("2023", 8553));
//        model.addAttribute("patientStats", patientStats);
//
//        // 模拟疾病统计数据
//        DiseaseStatistics diseaseStats = new DiseaseStatistics();
//        diseaseStats.setDiabeticRetinopathy(23);
//        diseaseStats.setGlaucoma(15);
//        diseaseStats.setMacularDegeneration(15);
//        diseaseStats.setCataract(10);
//        diseaseStats.setHypertensiveRetinopathy(8);
//        diseaseStats.setMyopia(26);
//        diseaseStats.setOther(3);
//        model.addAttribute("diseaseStats", diseaseStats);
//
//        return "statistics"; // 对应模板文件名(不带.html)
//    }
}