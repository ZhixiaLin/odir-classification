package com.example.logindemo.model;

public class PatientStatistics {
    private String year;
    private int count;

    // 构造函数、getter和setter
    public PatientStatistics(String year, int count) {
        this.year = year;
        this.count = count;
    }

    // getter和setter方法
    public String getYear() {
        return year;
    }

    public void setYear(String year) {
        this.year = year;
    }

    public int getCount() {
        return count;
    }

    public void setCount(int count) {
        this.count = count;
    }
}

