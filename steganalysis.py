#!/usr/bin/env python3
"""
Gelişmiş Steganografi Analiz Aracı
PNG, BMP ve JPEG dosyalarında çoklu yöntemle gizli veri tespiti
"""

import sys
import os
import struct
import zlib
from collections import Counter
import numpy as np
from PIL import Image
import argparse
import base64
from datetime import datetime
import json
import glob
from pathlib import Path

class AdvancedSteganalysisError(Exception):
    """Gelişmiş Steganaliz araç hatası"""
    pass

class AdvancedSteganalysisTool:
    def __init__(self, filepath, methods=None):
        self.filepath = filepath
        self.filename = os.path.basename(filepath)
        self.methods = methods or ['all']
        self.results = {
            'filename': self.filename,
            'file_type': None,
            'file_size': os.path.getsize(filepath),
            'analysis_methods': self.methods,
            'suspicious_findings': [],
            'extracted_data': [],
            'lsb_analysis': {},
            'dct_analysis': {},
            'histogram_analysis': {},
            'rs_analysis': {},
            'sample_pair_analysis': {},
            'entropy_analysis': {},
            'ml_prediction': {},
            'metadata': {},
            'chi_square_test': None
        }
        
    def analyze(self):
        """Ana analiz fonksiyonu - seçilen yöntemleri uygula"""
        print(f"\n{'='*70}")
        print(f"🔍 Gelişmiş Steganografi Analizi: {self.filename}")
        print(f"{'='*70}\n")
        
        # Dosya türünü belirle
        self._detect_file_type()
        
        # Görüntüyü yükle
        try:
            self.image = Image.open(self.filepath)
            self.image_array = np.array(self.image)
        except Exception as e:
            raise AdvancedSteganalysisError(f"Görüntü yüklenemedi: {e}")
        
        # Temel metadata
        self._analyze_metadata()
        
        # Seçilen analizleri yap
        if 'all' in self.methods or 'lsb' in self.methods:
            print("LSB Analizi...")
            self._analyze_lsb()
            self._chi_square_test()
            self._extract_lsb_data()
        
        if 'all' in self.methods or 'dct' in self.methods:
            if self.results['file_type'] == 'JPEG':
                print("DCT Analizi...")
                self._analyze_dct()
            else:
                print("DCT analizi sadece JPEG dosyaları için geçerli")
        
        if 'all' in self.methods or 'histogram' in self.methods:
            print("Histogram Analizi...")
            self._analyze_histogram()
        
        if 'all' in self.methods or 'rs' in self.methods:
            print("RS Analizi...")
            self._analyze_rs()
        
        if 'all' in self.methods or 'sample_pair' in self.methods:
            print("Sample Pair Analizi...")
            self._analyze_sample_pair()
        
        if 'all' in self.methods or 'entropy' in self.methods:
            print("Entropi Analizi...")
            self._analyze_entropy()
        
        if 'all' in self.methods or 'ml' in self.methods:
            print("Machine Learning Tespiti...")
            self._ml_based_detection()
        
        # Dosya yapısı (tüm analizlerde)
        self._analyze_file_structure()
        self._check_unusual_patterns()
        
        # Sonuçları göster
        self._display_results()
        
        return self.results
    
    def _detect_file_type(self):
        """Dosya türünü belirle"""
        with open(self.filepath, 'rb') as f:
            header = f.read(10)
        
        if header[:8] == b'\x89PNG\r\n\x1a\n':
            self.results['file_type'] = 'PNG'
        elif header[:2] == b'BM':
            self.results['file_type'] = 'BMP'
        elif header[:3] == b'\xff\xd8\xff':
            self.results['file_type'] = 'JPEG'
        else:
            self.results['file_type'] = 'UNKNOWN'
            self.results['suspicious_findings'].append(
                "Dosya başlığı beklenen formatta değil"
            )
    
    def _analyze_lsb(self):
        """LSB (Least Significant Bit) analizi"""
        if len(self.image_array.shape) == 3:
            channels = ['Red', 'Green', 'Blue']
            if self.image_array.shape[2] == 4:
                channels.append('Alpha')
            
            for idx, channel_name in enumerate(channels[:self.image_array.shape[2]]):
                channel = self.image_array[:, :, idx]
                lsb_bits = channel & 1
                
                ones = np.sum(lsb_bits)
                zeros = lsb_bits.size - ones
                ratio = ones / lsb_bits.size if lsb_bits.size > 0 else 0
                
                self.results['lsb_analysis'][channel_name] = {
                    'ones': int(ones),
                    'zeros': int(zeros),
                    'ratio': float(ratio),
                    'total_bits': int(lsb_bits.size)
                }
                
                if abs(ratio - 0.5) > 0.05:
                    self.results['suspicious_findings'].append(
                        f"{channel_name} kanalında anormal LSB dağılımı (Oran: {ratio:.3f})"
                    )
        else:
            lsb_bits = self.image_array & 1
            ones = np.sum(lsb_bits)
            zeros = lsb_bits.size - ones
            ratio = ones / lsb_bits.size if lsb_bits.size > 0 else 0
            
            self.results['lsb_analysis']['Grayscale'] = {
                'ones': int(ones),
                'zeros': int(zeros),
                'ratio': float(ratio),
                'total_bits': int(lsb_bits.size)
            }
            
            if abs(ratio - 0.5) > 0.05:
                self.results['suspicious_findings'].append(
                    f"Gri tonlamada anormal LSB dağılımı (Oran: {ratio:.3f})"
                )
    
    def _chi_square_test(self):
        """Chi-Square testi"""
        if len(self.image_array.shape) == 3:
            data = self.image_array[:, :, 0].flatten()
        else:
            data = self.image_array.flatten()
        
        pairs = {}
        for i in range(0, 256, 2):
            pairs[i] = np.sum(data == i)
            pairs[i+1] = np.sum(data == i+1)
        
        chi_square = 0
        for i in range(0, 256, 2):
            expected = (pairs[i] + pairs[i+1]) / 2
            if expected > 0:
                chi_square += ((pairs[i] - expected) ** 2) / expected
                chi_square += ((pairs[i+1] - expected) ** 2) / expected
        
        self.results['chi_square_test'] = float(chi_square)
        
        if chi_square > 200:
            self.results['suspicious_findings'].append(
                f"Yüksek Chi-Square değeri: {chi_square:.2f} (Muhtemel steganografi)"
            )
    
    def _analyze_dct(self):
        """DCT (Discrete Cosine Transform) analizi - JPEG için"""
        try:
            # JPEG DCT katsayılarını analiz et
            if self.results['file_type'] != 'JPEG':
                return
            
            # Piksel değerlerinden DCT tahmini yap
            if len(self.image_array.shape) == 3:
                gray = np.mean(self.image_array, axis=2).astype(np.float32)
            else:
                gray = self.image_array.astype(np.float32)
            
            # 8x8 bloklar halinde DCT analizi
            h, w = gray.shape
            h_blocks = h // 8
            w_blocks = w // 8
            
            dct_coeffs = []
            for i in range(h_blocks):
                for j in range(w_blocks):
                    block = gray[i*8:(i+1)*8, j*8:(j+1)*8]
                    # Basit DCT tahmini için varyans kullan
                    variance = np.var(block)
                    dct_coeffs.append(variance)
            
            dct_coeffs = np.array(dct_coeffs)
            
            self.results['dct_analysis'] = {
                'mean_variance': float(np.mean(dct_coeffs)),
                'std_variance': float(np.std(dct_coeffs)),
                'total_blocks': len(dct_coeffs),
                'suspicious_blocks': int(np.sum(dct_coeffs < 1.0))  # Çok düşük varyans
            }
            
            # Düşük varyans = muhtemel steganografi
            suspicious_ratio = self.results['dct_analysis']['suspicious_blocks'] / len(dct_coeffs)
            if suspicious_ratio > 0.1:
                self.results['suspicious_findings'].append(
                    f"DCT: %{suspicious_ratio*100:.1f} blokta düşük varyans (Muhtemel JPEG steganografi)"
                )
                
        except Exception as e:
            self.results['dct_analysis'] = {'error': str(e)}
    
    def _analyze_histogram(self):
        """Histogram analizi"""
        if len(self.image_array.shape) == 3:
            channels_data = [self.image_array[:, :, i] for i in range(self.image_array.shape[2])]
            channel_names = ['Red', 'Green', 'Blue', 'Alpha'][:self.image_array.shape[2]]
        else:
            channels_data = [self.image_array]
            channel_names = ['Grayscale']
        
        for channel_data, channel_name in zip(channels_data, channel_names):
            hist, bins = np.histogram(channel_data.flatten(), bins=256, range=(0, 256))
            
            # Histogram anomalileri tespit et
            # 1. Boş binler
            empty_bins = np.sum(hist == 0)
            
            # 2. Histogram düzgünlüğü (smoothness)
            hist_diff = np.abs(np.diff(hist))
            smoothness = np.mean(hist_diff)
            
            # 3. Spike tespiti
            threshold = np.mean(hist) + 3 * np.std(hist)
            spikes = np.sum(hist > threshold)
            
            self.results['histogram_analysis'][channel_name] = {
                'empty_bins': int(empty_bins),
                'smoothness': float(smoothness),
                'spikes': int(spikes),
                'mean_frequency': float(np.mean(hist)),
                'std_frequency': float(np.std(hist))
            }
            
            # Anomali kontrolü
            if empty_bins > 50:
                self.results['suspicious_findings'].append(
                    f"Histogram: {channel_name} kanalında {empty_bins} boş bin (Veri manipülasyonu olabilir)"
                )
            
            if spikes > 5:
                self.results['suspicious_findings'].append(
                    f"Histogram: {channel_name} kanalında {spikes} anormal spike"
                )
    
    def _analyze_rs(self):
        """RS (Regular-Singular) analizi"""
        try:
            if len(self.image_array.shape) == 3:
                data = self.image_array[:, :, 0].flatten()
            else:
                data = self.image_array.flatten()
            
            # Mask fonksiyonları
            def flip_lsb(x):
                return x ^ 1
            
            def regular_groups(arr):
                # Regular: komşu pikseller arasındaki fark değişmez
                return np.sum(np.abs(np.diff(arr)))
            
            # Original
            original = data[:len(data)//2 * 2].reshape(-1, 2)
            R_original = 0
            S_original = 0
            
            for group in original:
                f = regular_groups(group)
                flipped = flip_lsb(group)
                f_flipped = regular_groups(flipped)
                
                if f > f_flipped:
                    R_original += 1
                elif f < f_flipped:
                    S_original += 1
            
            # Negatif mask
            flipped_data = flip_lsb(data)
            flipped = flipped_data[:len(flipped_data)//2 * 2].reshape(-1, 2)
            R_flipped = 0
            S_flipped = 0
            
            for group in flipped:
                f = regular_groups(group)
                reflipped = flip_lsb(group)
                f_reflipped = regular_groups(reflipped)
                
                if f > f_reflipped:
                    R_flipped += 1
                elif f < f_reflipped:
                    S_flipped += 1
            
            total_groups = len(original)
            
            self.results['rs_analysis'] = {
                'R_original': int(R_original),
                'S_original': int(S_original),
                'R_flipped': int(R_flipped),
                'S_flipped': int(S_flipped),
                'total_groups': int(total_groups),
                'R_ratio': float(R_original / total_groups) if total_groups > 0 else 0,
                'S_ratio': float(S_original / total_groups) if total_groups > 0 else 0
            }
            
            # RS anomali tespiti
            # Normal görüntülerde R ≈ S olmalı
            rs_diff = abs(R_original - S_original) / total_groups if total_groups > 0 else 0
            
            if rs_diff > 0.05:
                embedded_ratio = rs_diff  # Tahmini gömülü veri oranı
                self.results['suspicious_findings'].append(
                    f"RS Analizi: R-S farkı yüksek (Tahmini gömülü veri: %{embedded_ratio*100:.1f})"
                )
                
        except Exception as e:
            self.results['rs_analysis'] = {'error': str(e)}
    
    def _analyze_sample_pair(self):
        """Sample Pair analizi"""
        try:
            if len(self.image_array.shape) == 3:
                data = self.image_array[:, :, 0].flatten()
            else:
                data = self.image_array.flatten()
            
            # Komşu piksel çiftlerini analiz et
            pairs = []
            for i in range(0, len(data)-1, 2):
                pairs.append((int(data[i]), int(data[i+1])))
            
            # Çift türlerini say
            X = 0  # u = 2k, v = 2k
            Y = 0  # u = 2k+1, v = 2k+1
            Z = 0  # u = 2k, v = 2k+1 or vice versa
            
            for u, v in pairs:
                if u % 2 == 0 and v % 2 == 0:
                    X += 1
                elif u % 2 == 1 and v % 2 == 1:
                    Y += 1
                else:
                    Z += 1
            
            total = len(pairs)
            
            # Tahmini gömülü mesaj uzunluğu
            # LSB embedding'de X ve Y azalır, Z artar
            expected_X = total * 0.25
            expected_Y = total * 0.25
            expected_Z = total * 0.5
            
            deviation_X = abs(X - expected_X) / expected_X if expected_X > 0 else 0
            deviation_Z = abs(Z - expected_Z) / expected_Z if expected_Z > 0 else 0
            
            self.results['sample_pair_analysis'] = {
                'X_pairs': int(X),
                'Y_pairs': int(Y),
                'Z_pairs': int(Z),
                'total_pairs': int(total),
                'X_ratio': float(X / total) if total > 0 else 0,
                'Y_ratio': float(Y / total) if total > 0 else 0,
                'Z_ratio': float(Z / total) if total > 0 else 0,
                'deviation_score': float(deviation_X + deviation_Z)
            }
            
            # Anomali kontrolü
            if deviation_X > 0.15 or deviation_Z > 0.15:
                self.results['suspicious_findings'].append(
                    f"Sample Pair: Anormal çift dağılımı (Sapma skoru: {deviation_X + deviation_Z:.3f})"
                )
                
        except Exception as e:
            self.results['sample_pair_analysis'] = {'error': str(e)}
    
    def _analyze_entropy(self):
        """Entropi analizi - Shannon entropisi"""
        try:
            if len(self.image_array.shape) == 3:
                channels_data = [self.image_array[:, :, i] for i in range(self.image_array.shape[2])]
                channel_names = ['Red', 'Green', 'Blue', 'Alpha'][:self.image_array.shape[2]]
            else:
                channels_data = [self.image_array]
                channel_names = ['Grayscale']
            
            for channel_data, channel_name in zip(channels_data, channel_names):
                flat = channel_data.flatten()
                
                # Shannon entropisi
                hist, _ = np.histogram(flat, bins=256, range=(0, 256))
                hist = hist[hist > 0]  # Sıfırları kaldır
                probabilities = hist / len(flat)
                entropy = -np.sum(probabilities * np.log2(probabilities))
                
                # LSB entropisi (sadece LSB bitlerinin entropisi)
                lsb_bits = flat & 1
                lsb_hist = np.bincount(lsb_bits, minlength=2)
                lsb_prob = lsb_hist / len(lsb_bits)
                lsb_prob = lsb_prob[lsb_prob > 0]
                lsb_entropy = -np.sum(lsb_prob * np.log2(lsb_prob))
                
                # Blok bazlı entropi varyansı
                h, w = channel_data.shape
                block_size = 32
                h_blocks = h // block_size
                w_blocks = w // block_size
                
                block_entropies = []
                for i in range(h_blocks):
                    for j in range(w_blocks):
                        block = channel_data[i*block_size:(i+1)*block_size, 
                                            j*block_size:(j+1)*block_size].flatten()
                        b_hist, _ = np.histogram(block, bins=256, range=(0, 256))
                        b_hist = b_hist[b_hist > 0]
                        b_prob = b_hist / len(block)
                        b_entropy = -np.sum(b_prob * np.log2(b_prob))
                        block_entropies.append(b_entropy)
                
                block_entropies = np.array(block_entropies)
                
                self.results['entropy_analysis'][channel_name] = {
                    'shannon_entropy': float(entropy),
                    'lsb_entropy': float(lsb_entropy),
                    'mean_block_entropy': float(np.mean(block_entropies)),
                    'std_block_entropy': float(np.std(block_entropies)),
                    'max_entropy': 8.0  # 256 değer için maksimum
                }
                
                # Anomali kontrolü
                # Yüksek entropi = rastgele/şifreli veri
                if entropy > 7.5:
                    self.results['suspicious_findings'].append(
                        f"Entropi: {channel_name} kanalında çok yüksek entropi ({entropy:.2f}) - Şifreli veri olabilir"
                    )
                
                # LSB entropisi 1'e yakın = ideal rastgele
                if lsb_entropy > 0.99:
                    self.results['suspicious_findings'].append(
                        f"Entropi: {channel_name} LSB entropisi çok yüksek ({lsb_entropy:.3f}) - Steganografi olabilir"
                    )
                
        except Exception as e:
            self.results['entropy_analysis'] = {'error': str(e)}
    
    def _ml_based_detection(self):
        """Machine Learning tabanlı tespit - Feature-based"""
        try:
            # Özellik çıkarımı
            features = []
            
            # 1. LSB özellikleri
            if self.results['lsb_analysis']:
                for channel, stats in self.results['lsb_analysis'].items():
                    features.append(stats['ratio'])
            
            # 2. Chi-square
            if self.results['chi_square_test']:
                features.append(self.results['chi_square_test'] / 1000)  # Normalize
            
            # 3. Histogram özellikleri
            if self.results['histogram_analysis']:
                for channel, stats in self.results['histogram_analysis'].items():
                    features.append(stats['smoothness'] / 100)
                    features.append(stats['empty_bins'] / 256)
            
            # 4. Entropi özellikleri
            if self.results['entropy_analysis']:
                for channel, stats in self.results['entropy_analysis'].items():
                    features.append(stats['shannon_entropy'] / 8)
                    features.append(stats['lsb_entropy'])
            
            # 5. RS özellikleri
            if self.results['rs_analysis'] and 'error' not in self.results['rs_analysis']:
                features.append(self.results['rs_analysis']['R_ratio'])
                features.append(self.results['rs_analysis']['S_ratio'])
            
            # 6. Sample Pair özellikleri
            if self.results['sample_pair_analysis'] and 'error' not in self.results['sample_pair_analysis']:
                features.append(self.results['sample_pair_analysis']['deviation_score'])
            
            # Basit ML skor hesaplama (weighted sum)
            # Gerçek ML için sklearn kullanılabilir
            weights = {
                'lsb_deviation': 15,
                'chi_square': 20,
                'histogram_anomaly': 10,
                'entropy_high': 15,
                'rs_anomaly': 20,
                'sample_pair_anomaly': 10,
                'findings_count': 10
            }
            
            ml_score = 0
            
            # LSB sapması
            if self.results['lsb_analysis']:
                for stats in self.results['lsb_analysis'].values():
                    lsb_dev = abs(stats['ratio'] - 0.5)
                    if lsb_dev > 0.05:
                        ml_score += weights['lsb_deviation'] * (lsb_dev / 0.5)
            
            # Chi-square
            if self.results['chi_square_test']:
                if self.results['chi_square_test'] > 100:
                    ml_score += weights['chi_square']
            
            # Bulgular
            ml_score += len(self.results['suspicious_findings']) * weights['findings_count']
            
            # Normalize (0-100)
            ml_score = min(100, ml_score)
            
            # Güven aralığı
            confidence = "Düşük"
            if ml_score > 70:
                confidence = "Yüksek"
            elif ml_score > 40:
                confidence = "Orta"
            
            self.results['ml_prediction'] = {
                'score': float(ml_score),
                'confidence': confidence,
                'prediction': 'Steganografi Tespit Edildi' if ml_score > 50 else 'Temiz',
                'features_used': len(features),
                'risk_level': 'YÜKSEK' if ml_score > 70 else ('ORTA' if ml_score > 40 else 'DÜŞÜK')
            }
            
            if ml_score > 60:
                self.results['suspicious_findings'].append(
                    f"ML Tespiti: Yüksek steganografi skoru ({ml_score:.1f}/100) - {confidence} güven"
                )
                
        except Exception as e:
            self.results['ml_prediction'] = {'error': str(e)}
    
    def _extract_lsb_data(self):
        """LSB'lerden veri çıkar"""
        if len(self.image_array.shape) == 3:
            flat_image = self.image_array[:, :, :3].flatten()
        else:
            flat_image = self.image_array.flatten()
        
        lsb_bits = (flat_image & 1).astype(np.uint8)
        
        bytes_data = []
        for i in range(0, len(lsb_bits) - 8, 8):
            byte = 0
            for j in range(8):
                byte |= (lsb_bits[i + j] << j)
            bytes_data.append(byte)
        
        text_attempt = self._try_extract_text(bytes_data)
        if text_attempt:
            self.results['extracted_data'].append({
                'type': 'ASCII Text (LSB)',
                'data': text_attempt,
                'length': len(text_attempt)
            })
            self.results['suspicious_findings'].append(
                f"LSB'lerde ASCII metin bulundu! ({len(text_attempt)} karakter)"
            )
        
        file_signatures = self._check_file_signatures(bytes_data)
        if file_signatures:
            self.results['extracted_data'].extend(file_signatures)
    
    def _try_extract_text(self, bytes_data, min_length=10):
        """Byte dizisinden ASCII metin çıkar"""
        text = ""
        consecutive_printable = 0
        
        for byte in bytes_data[:10000]:
            if 32 <= byte <= 126 or byte in [9, 10, 13]:
                text += chr(byte)
                consecutive_printable += 1
            else:
                if consecutive_printable >= min_length:
                    return text
                text = ""
                consecutive_printable = 0
        
        if len(text) >= min_length:
            return text
        return None
    
    def _check_file_signatures(self, bytes_data):
        """Dosya imzalarını kontrol et"""
        signatures = {
            'PNG': [b'\x89PNG\r\n\x1a\n'],
            'JPEG': [b'\xFF\xD8\xFF'],
            'ZIP': [b'PK\x03\x04', b'PK\x05\x06'],
            'PDF': [b'%PDF'],
            'GIF': [b'GIF87a', b'GIF89a'],
            'RAR': [b'Rar!\x1a\x07'],
            'EXE': [b'MZ'],
            'MP3': [b'ID3', b'\xFF\xFB'],
        }
        
        found_files = []
        bytes_array = bytes(bytes_data[:1000])
        
        for file_type, sigs in signatures.items():
            for sig in sigs:
                if sig in bytes_array:
                    found_files.append({
                        'type': f'File Signature: {file_type}',
                        'data': f'Offset: {bytes_array.find(sig)}',
                        'length': len(sig)
                    })
                    self.results['suspicious_findings'].append(
                        f"{file_type} dosya imzası tespit edildi!"
                    )
        
        return found_files
    
    def _check_unusual_patterns(self):
        """Olağandışı paternleri kontrol et"""
        std_dev = np.std(self.image_array)
        if std_dev < 10:
            self.results['suspicious_findings'].append(
                f"Çok düşük standart sapma: {std_dev:.2f}"
            )
        
        flat = self.image_array.flatten()
        sequential_count = 0
        for i in range(len(flat) - 1):
            if abs(int(flat[i]) - int(flat[i+1])) <= 1:
                sequential_count += 1
        
        sequential_ratio = sequential_count / len(flat)
        if sequential_ratio > 0.8:
            self.results['suspicious_findings'].append(
                f"Yüksek sıralı değer oranı: {sequential_ratio:.2%}"
            )
    
    def _analyze_metadata(self):
        """Metadata analizi"""
        try:
            from PIL.ExifTags import TAGS
            exif = self.image._getexif()
            if exif:
                for tag_id, value in exif.items():
                    tag = TAGS.get(tag_id, tag_id)
                    self.results['metadata'][tag] = str(value)
        except:
            pass
        
        self.results['metadata']['Format'] = self.image.format
        self.results['metadata']['Mode'] = self.image.mode
        self.results['metadata']['Size'] = f"{self.image.size[0]}x{self.image.size[1]}"
        
        if self.results['file_type'] == 'PNG':
            self._analyze_png_chunks()
    
    def _analyze_png_chunks(self):
        """PNG chunk analizi"""
        with open(self.filepath, 'rb') as f:
            f.read(8)
            
            chunks = []
            while True:
                try:
                    length_bytes = f.read(4)
                    if len(length_bytes) < 4:
                        break
                    
                    length = struct.unpack('>I', length_bytes)[0]
                    chunk_type = f.read(4).decode('ascii', errors='ignore')
                    chunk_data = f.read(length)
                    crc = f.read(4)
                    
                    chunks.append({'type': chunk_type, 'length': length})
                    
                    standard_chunks = ['IHDR', 'PLTE', 'IDAT', 'IEND', 'tRNS', 
                                      'gAMA', 'cHRM', 'sRGB', 'iCCP', 'tEXt', 
                                      'zTXt', 'iTXt', 'bKGD', 'pHYs', 'tIME']
                    
                    if chunk_type not in standard_chunks:
                        self.results['suspicious_findings'].append(
                            f"Standart olmayan PNG chunk: {chunk_type} ({length} byte)"
                        )
                        
                except Exception:
                    break
            
            self.results['metadata']['PNG_Chunks'] = chunks
    
    def _analyze_file_structure(self):
        """Dosya yapısı analizi"""
        with open(self.filepath, 'rb') as f:
            file_data = f.read()
        
        if self.results['file_type'] == 'PNG':
            iend_pos = file_data.rfind(b'IEND')
            if iend_pos != -1:
                expected_end = iend_pos + 4 + 4
                if len(file_data) > expected_end:
                    extra_bytes = len(file_data) - expected_end
                    self.results['suspicious_findings'].append(
                        f"PNG IEND chunk'ından sonra {extra_bytes} byte ekstra veri!"
                    )
                    
                    extra_data = file_data[expected_end:]
                    text = self._try_extract_text(list(extra_data))
                    if text:
                        self.results['extracted_data'].append({
                            'type': 'Text after IEND',
                            'data': text,
                            'length': len(text)
                        })
    
    def _display_results(self):
        """Sonuçları göster"""
        print(f"\n{'='*70}")
        print("ANALİZ SONUÇLARI")
        print(f"{'='*70}\n")
        
        print(f"Dosya: {self.results['filename']}")
        print(f"Tip: {self.results['file_type']}")
        print(f"Boyut: {self.results['file_size']:,} bytes")
        print(f"Uygulanan Yöntemler: {', '.join(self.results['analysis_methods'])}")
        
        # ML Skoru (varsa)
        if self.results['ml_prediction'] and 'score' in self.results['ml_prediction']:
            ml = self.results['ml_prediction']
            print(f"\n{'─'*70}")
            print(f"MACHINE LEARNING TAHMİNİ")
            print(f"{'─'*70}")
            print(f"Skor: {ml['score']:.1f}/100")
            print(f"Risk Seviyesi: {ml['risk_level']}")
            print(f"Tahmin: {ml['prediction']}")
            print(f"Güven: {ml['confidence']}")
        
        # Diğer analizler...
        if self.results['lsb_analysis']:
            print(f"\n{'─'*70}")
            print("LSB Analizi:")
            for channel, stats in self.results['lsb_analysis'].items():
                print(f"  {channel}: 1'ler {stats['ratio']:.1%}, Chi²: {self.results.get('chi_square_test', 0):.1f}")
        
        if self.results['entropy_analysis']:
            print(f"\n{'─'*70}")
            print("🌡️  Entropi Analizi:")
            for channel, stats in self.results['entropy_analysis'].items():
                print(f"  {channel}: Shannon={stats['shannon_entropy']:.2f}, LSB={stats['lsb_entropy']:.3f}")
        
        if self.results['rs_analysis'] and 'error' not in self.results['rs_analysis']:
            print(f"\n{'─'*70}")
            print(f"🔬 RS Analizi: R={self.results['rs_analysis']['R_ratio']:.3f}, S={self.results['rs_analysis']['S_ratio']:.3f}")
        
        if self.results['suspicious_findings']:
            print(f"\n{'─'*70}")
            print(f"ŞÜPHELİ BULGULAR ({len(self.results['suspicious_findings'])}):")
            print(f"{'─'*70}")
            for finding in self.results['suspicious_findings'][:10]:  # İlk 10 bulgu
                print(f"  {finding}")
            if len(self.results['suspicious_findings']) > 10:
                print(f"  ... ve {len(self.results['suspicious_findings']) - 10} bulgu daha")
        
        if self.results['extracted_data']:
            print(f"\n{'─'*70}")
            print(f"ÇIKARILAN VERİLER ({len(self.results['extracted_data'])}):")
            print(f"{'─'*70}")
            for idx, data in enumerate(self.results['extracted_data'][:5], 1):
                print(f"\n  [{idx}] {data['type']} ({data['length']} bytes)")
                if isinstance(data['data'], str) and len(data['data']) <= 200:
                    print(f"      {data['data'][:200]}")
        
        print(f"\n{'='*70}")
        risk_score = self.results['ml_prediction'].get('score', 0) if self.results['ml_prediction'] else 0
        if risk_score > 50 or self.results['suspicious_findings']:
            print("SONUÇ: Bu dosyada steganografi belirtileri tespit edildi!")
        else:
            print("SONUÇ: Belirgin bir steganografi tespit edilemedi.")
        print(f"{'='*70}\n")
    
    def generate_html_report(self, output_path='report.html'):
        """Kapsamlı HTML raporu oluştur - Tüm analizleri içerir"""
        
        with open(self.filepath, 'rb') as f:
            image_data = base64.b64encode(f.read()).decode('utf-8')
        
        analysis_time = datetime.now().strftime('%d.%m.%Y %H:%M:%S')
        
        # Risk skoru hesapla
        ml_score = self.results['ml_prediction'].get('score', 0) if self.results['ml_prediction'] else 0
        risk_score = ml_score if ml_score > 0 else (
            min(100, len(self.results['suspicious_findings']) * 10 + 
                (self.results.get('chi_square_test', 0) / 10 if self.results.get('chi_square_test') else 0))
        )
        
        if risk_score >= 70:
            risk_level = "YÜKSEK RİSK"
            risk_color = "#ef4444"
            risk_bg = "#fef2f2"
        elif risk_score >= 40:
            risk_level = "ORTA RİSK"
            risk_color = "#f59e0b"
            risk_bg = "#fffbeb"
        else:
            risk_level = "DÜŞÜK RİSK"
            risk_color = "#10b981"
            risk_bg = "#f0fdf4"
        
        html_content = f"""<!DOCTYPE html>
<html lang="tr">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Gelişmiş Steganografi Raporu - {self.results['filename']}</title>
    <style>
        @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;700&family=Outfit:wght@300;400;600;700&display=swap');
        
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        
        :root {{
            --bg-primary: #0a0e27;
            --bg-secondary: #111827;
            --bg-card: #1a1f3a;
            --accent: #6366f1;
            --accent-bright: #818cf8;
            --success: #10b981;
            --warning: #f59e0b;
            --danger: #ef4444;
            --text: #f1f5f9;
            --text-dim: #94a3b8;
            --border: #2d3748;
        }}
        
        body {{
            font-family: 'Outfit', sans-serif;
            background: linear-gradient(135deg, var(--bg-primary) 0%, var(--bg-secondary) 100%);
            color: var(--text);
            line-height: 1.6;
            min-height: 100vh;
        }}
        
        .container {{ max-width: 1400px; margin: 0 auto; padding: 40px 20px; }}
        
        .header {{
            text-align: center;
            padding: 50px 30px;
            background: rgba(26, 31, 58, 0.8);
            backdrop-filter: blur(15px);
            border-radius: 25px;
            margin-bottom: 40px;
            border: 1px solid var(--border);
            position: relative;
            overflow: hidden;
        }}
        
        .header::before {{
            content: '';
            position: absolute;
            top: 0; left: 0; right: 0;
            height: 4px;
            background: linear-gradient(90deg, var(--accent), var(--accent-bright), var(--accent));
            background-size: 200% 100%;
            animation: shimmer 3s linear infinite;
        }}
        
        @keyframes shimmer {{ 0% {{ background-position: -200% 0; }} 100% {{ background-position: 200% 0; }} }}
        
        .header h1 {{
            font-size: 2.8em;
            font-weight: 700;
            margin-bottom: 15px;
            background: linear-gradient(135deg, var(--accent) 0%, var(--accent-bright) 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }}
        
        .risk-banner {{
            background: {risk_bg};
            border: 2px solid {risk_color};
            border-radius: 20px;
            padding: 35px;
            margin: 30px 0;
            text-align: center;
        }}
        
        .risk-score {{
            font-size: 4.5em;
            font-weight: 700;
            color: {risk_color};
            font-family: 'JetBrains Mono', monospace;
        }}
        
        .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(350px, 1fr)); gap: 25px; margin: 30px 0; }}
        
        .card {{
            background: rgba(26, 31, 58, 0.9);
            backdrop-filter: blur(10px);
            border-radius: 20px;
            padding: 30px;
            border: 1px solid var(--border);
            transition: all 0.3s ease;
        }}
        
        .card:hover {{
            transform: translateY(-5px);
            box-shadow: 0 20px 60px rgba(99, 102, 241, 0.3);
            border-color: var(--accent);
        }}
        
        .card h2 {{
            font-size: 1.5em;
            margin-bottom: 20px;
            color: var(--accent-bright);
            display: flex;
            align-items: center;
            gap: 12px;
        }}
        
        .chart-container {{
            margin: 20px 0;
            padding: 20px;
            background: rgba(10, 14, 39, 0.5);
            border-radius: 12px;
        }}
        
        .bar {{
            display: flex;
            align-items: center;
            margin: 15px 0;
        }}
        
        .bar-label {{
            width: 120px;
            font-weight: 600;
            font-family: 'JetBrains Mono', monospace;
            font-size: 0.9em;
        }}
        
        .bar-container {{
            flex: 1;
            height: 32px;
            background: rgba(10, 14, 39, 0.8);
            border-radius: 6px;
            overflow: hidden;
            position: relative;
        }}
        
        .bar-fill {{
            height: 100%;
            background: linear-gradient(90deg, var(--accent), var(--accent-bright));
            display: flex;
            align-items: center;
            justify-content: flex-end;
            padding: 0 12px;
            font-family: 'JetBrains Mono', monospace;
            font-weight: 700;
            font-size: 0.85em;
            transition: width 1.2s cubic-bezier(0.4, 0, 0.2, 1);
        }}
        
        .metric-box {{
            background: rgba(10, 14, 39, 0.6);
            padding: 20px;
            border-radius: 12px;
            margin: 15px 0;
            border-left: 4px solid var(--accent);
        }}
        
        .metric-value {{
            font-size: 2em;
            font-weight: 700;
            color: var(--accent-bright);
            font-family: 'JetBrains Mono', monospace;
        }}
        
        .metric-label {{
            color: var(--text-dim);
            font-size: 0.9em;
            margin-top: 5px;
        }}
        
        .finding-item {{
            background: rgba(10, 14, 39, 0.5);
            padding: 15px;
            border-radius: 10px;
            margin: 12px 0;
            border-left: 4px solid var(--warning);
            font-family: 'JetBrains Mono', monospace;
            font-size: 0.9em;
        }}
        
        .finding-item.critical {{ border-left-color: var(--danger); }}
        .finding-item.success {{ border-left-color: var(--success); }}
        
        .badge {{
            display: inline-block;
            padding: 6px 14px;
            border-radius: 20px;
            font-size: 0.8em;
            font-weight: 600;
            font-family: 'JetBrains Mono', monospace;
        }}
        
        .badge-success {{ background: rgba(16, 185, 129, 0.2); color: var(--success); border: 1px solid var(--success); }}
        .badge-warning {{ background: rgba(245, 158, 11, 0.2); color: var(--warning); border: 1px solid var(--warning); }}
        .badge-danger {{ background: rgba(239, 68, 68, 0.2); color: var(--danger); border: 1px solid var(--danger); }}
        
        .info-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 15px; }}
        .info-item {{ background: rgba(10, 14, 39, 0.5); padding: 15px; border-radius: 10px; }}
        .info-item strong {{ display: block; color: var(--text-dim); font-size: 0.85em; margin-bottom: 5px; }}
        .info-item span {{ font-family: 'JetBrains Mono', monospace; font-weight: 600; }}
        
        .image-preview {{ width: 100%; border-radius: 12px; margin-top: 15px; border: 2px solid var(--border); }}
        
        .data-extract {{
            background: rgba(10, 14, 39, 0.8);
            padding: 20px;
            border-radius: 12px;
            margin: 15px 0;
            border: 1px solid var(--border);
        }}
        
        .data-extract h3 {{ color: var(--accent); margin-bottom: 10px; }}
        
        .data-extract pre {{
            background: rgba(0, 0, 0, 0.6);
            padding: 15px;
            border-radius: 8px;
            overflow-x: auto;
            font-family: 'JetBrains Mono', monospace;
            font-size: 0.85em;
            line-height: 1.6;
            color: #94a3b8;
        }}
        
        .footer {{
            text-align: center;
            margin-top: 50px;
            padding: 30px;
            background: rgba(26, 31, 58, 0.6);
            border-radius: 20px;
            color: var(--text-dim);
        }}
        
        @media print {{ body {{ background: white; color: black; }} }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>Gelişmiş Steganografi Analizi</h1>
            <div style="color: var(--text-dim); font-size: 1.1em; font-family: 'JetBrains Mono', monospace;">
                {', '.join(self.results['analysis_methods'])} Analysis
            </div>
        </div>
        
        <div class="risk-banner">
            <div class="risk-score">{risk_score:.0f}</div>
            <div style="font-size: 1.8em; font-weight: 600; color: {risk_color}; margin-top: 10px;">{risk_level}</div>
            <p style="margin-top: 15px; color: {risk_color}; font-weight: 600;">
                {'Steganografi belirtileri tespit edildi!' if risk_score >= 40 else 'Dosya temiz görünüyor.'}
            </p>
        </div>
        
        <div class="grid">
            <div class="card">
                <h2>Dosya Bilgileri</h2>
                <div class="info-grid">
                    <div class="info-item"><strong>Dosya</strong><span>{self.results['filename']}</span></div>
                    <div class="info-item"><strong>Tip</strong><span>{self.results['file_type']}</span></div>
                    <div class="info-item"><strong>Boyut</strong><span>{self.results['file_size']:,} B</span></div>
                    <div class="info-item"><strong>Boyutlar</strong><span>{self.results['metadata'].get('Size', 'N/A')}</span></div>
                    <div class="info-item"><strong>Analiz</strong><span>{analysis_time}</span></div>
                </div>
            </div>
            
            <div class="card">
                <h2>Görüntü</h2>
                <img src="data:image/png;base64,{image_data}" alt="Analyzed image" class="image-preview">
            </div>
        </div>
"""
        
        # ML Prediction
        if self.results['ml_prediction'] and 'score' in self.results['ml_prediction']:
            ml = self.results['ml_prediction']
            html_content += f"""
        <div class="card">
            <h2>Machine Learning Tespiti</h2>
            <div class="metric-box" style="border-left-color: {risk_color};">
                <div class="metric-value" style="color: {risk_color};">{ml['score']:.1f}/100</div>
                <div class="metric-label">Steganografi Skoru</div>
            </div>
            <div class="info-grid" style="margin-top: 20px;">
                <div class="info-item"><strong>Tahmin</strong><span>{ml['prediction']}</span></div>
                <div class="info-item"><strong>Güven</strong><span>{ml['confidence']}</span></div>
                <div class="info-item"><strong>Risk</strong><span>{ml['risk_level']}</span></div>
                <div class="info-item"><strong>Özellik</strong><span>{ml['features_used']} features</span></div>
            </div>
        </div>
"""
        
        # LSB Analysis
        if self.results['lsb_analysis']:
            html_content += """
        <div class="card">
            <h2>LSB Analizi</h2>
            <div class="chart-container">
"""
            for channel, stats in self.results['lsb_analysis'].items():
                ratio_percent = stats['ratio'] * 100
                html_content += f"""
                <div class="bar">
                    <div class="bar-label">{channel}</div>
                    <div class="bar-container">
                        <div class="bar-fill" style="width: {ratio_percent}%">{ratio_percent:.1f}%</div>
                    </div>
                </div>
"""
            
            if self.results.get('chi_square_test'):
                chi = self.results['chi_square_test']
                chi_badge = 'badge-danger' if chi > 200 else ('badge-warning' if chi > 100 else 'badge-success')
                html_content += f"""
            </div>
            <div class="metric-box" style="margin-top: 20px;">
                <div class="metric-value">{chi:.1f}</div>
                <div class="metric-label">Chi-Square Test <span class="badge {chi_badge}">
                    {'YÜKSEK' if chi > 200 else ('ORTA' if chi > 100 else 'DÜŞÜK')}
                </span></div>
            </div>
"""
            html_content += """
        </div>
"""
        
        # Entropy Analysis
        if self.results['entropy_analysis']:
            html_content += """
        <div class="card">
            <h2>Entropi Analizi</h2>
"""
            for channel, stats in self.results['entropy_analysis'].items():
                if 'error' not in stats:
                    html_content += f"""
            <div class="metric-box">
                <div style="font-weight: 600; color: var(--accent); margin-bottom: 10px;">{channel}</div>
                <div class="info-grid">
                    <div class="info-item"><strong>Shannon</strong><span>{stats['shannon_entropy']:.3f}</span></div>
                    <div class="info-item"><strong>LSB</strong><span>{stats['lsb_entropy']:.3f}</span></div>
                    <div class="info-item"><strong>Mean Block</strong><span>{stats['mean_block_entropy']:.3f}</span></div>
                    <div class="info-item"><strong>Std Block</strong><span>{stats['std_block_entropy']:.3f}</span></div>
                </div>
            </div>
"""
            html_content += """
        </div>
"""
        
        # RS Analysis
        if self.results['rs_analysis'] and 'error' not in self.results['rs_analysis']:
            rs = self.results['rs_analysis']
            html_content += f"""
        <div class="card">
            <h2>🔬 RS Analizi</h2>
            <div class="info-grid">
                <div class="info-item"><strong>R Original</strong><span>{rs['R_original']}</span></div>
                <div class="info-item"><strong>S Original</strong><span>{rs['S_original']}</span></div>
                <div class="info-item"><strong>R Flipped</strong><span>{rs['R_flipped']}</span></div>
                <div class="info-item"><strong>S Flipped</strong><span>{rs['S_flipped']}</span></div>
            </div>
            <div class="chart-container" style="margin-top: 20px;">
                <div class="bar">
                    <div class="bar-label">R Ratio</div>
                    <div class="bar-container">
                        <div class="bar-fill" style="width: {rs['R_ratio']*100}%">{rs['R_ratio']:.3f}</div>
                    </div>
                </div>
                <div class="bar">
                    <div class="bar-label">S Ratio</div>
                    <div class="bar-container">
                        <div class="bar-fill" style="width: {rs['S_ratio']*100}%">{rs['S_ratio']:.3f}</div>
                    </div>
                </div>
            </div>
        </div>
"""
        
        # Sample Pair Analysis
        if self.results['sample_pair_analysis'] and 'error' not in self.results['sample_pair_analysis']:
            sp = self.results['sample_pair_analysis']
            html_content += f"""
        <div class="card">
            <h2>👥 Sample Pair Analizi</h2>
            <div class="info-grid">
                <div class="info-item"><strong>X Pairs</strong><span>{sp['X_pairs']}</span></div>
                <div class="info-item"><strong>Y Pairs</strong><span>{sp['Y_pairs']}</span></div>
                <div class="info-item"><strong>Z Pairs</strong><span>{sp['Z_pairs']}</span></div>
                <div class="info-item"><strong>Deviation</strong><span>{sp['deviation_score']:.3f}</span></div>
            </div>
        </div>
"""
        
        # Histogram Analysis
        if self.results['histogram_analysis']:
            html_content += """
        <div class="card">
            <h2>Histogram Analizi</h2>
"""
            for channel, stats in self.results['histogram_analysis'].items():
                html_content += f"""
            <div class="metric-box">
                <div style="font-weight: 600; color: var(--accent); margin-bottom: 10px;">{channel}</div>
                <div class="info-grid">
                    <div class="info-item"><strong>Empty Bins</strong><span>{stats['empty_bins']}</span></div>
                    <div class="info-item"><strong>Smoothness</strong><span>{stats['smoothness']:.1f}</span></div>
                    <div class="info-item"><strong>Spikes</strong><span>{stats['spikes']}</span></div>
                    <div class="info-item"><strong>Std</strong><span>{stats['std_frequency']:.1f}</span></div>
                </div>
            </div>
"""
            html_content += """
        </div>
"""
        
        # Suspicious Findings
        if self.results['suspicious_findings']:
            html_content += f"""
        <div class="card">
            <h2>Şüpheli Bulgular ({len(self.results['suspicious_findings'])})</h2>
"""
            for finding in self.results['suspicious_findings'][:20]:
                finding_class = 'success' if '✅' in finding else ('critical' if '🤖' in finding or 'YÜKSEK' in finding else '')
                html_content += f'            <div class="finding-item {finding_class}">{finding}</div>\n'
            
            if len(self.results['suspicious_findings']) > 20:
                html_content += f'            <p style="color: var(--text-dim); margin-top: 15px; text-align: center;">...ve {len(self.results["suspicious_findings"]) - 20} bulgu daha</p>\n'
            
            html_content += """
        </div>
"""
        
        # Extracted Data
        if self.results['extracted_data']:
            html_content += f"""
        <div class="card">
            <h2>Çıkarılan Veriler ({len(self.results['extracted_data'])})</h2>
"""
            for idx, data in enumerate(self.results['extracted_data'][:10], 1):
                data_content = str(data['data'])[:1000]
                html_content += f"""
            <div class="data-extract">
                <h3>[{idx}] {data['type']}</h3>
                <p><strong>Uzunluk:</strong> {data['length']} bytes</p>
                <pre>{data_content}</pre>
            </div>
"""
            html_content += """
        </div>
"""
        
        html_content += f"""
        <div class="footer">
            <p><strong>Gelişmiş Steganografi Analiz Aracı</strong></p>
            <p>{analysis_time} • Analiz Yöntemleri: {', '.join(self.results['analysis_methods'])}</p>
            <p style="margin-top: 15px; font-size: 0.85em;">
                Bu araç eğitim ve güvenlik araştırması amaçlıdır
            </p>
        </div>
    </div>
    
    <script>
        window.addEventListener('load', function() {{
            const bars = document.querySelectorAll('.bar-fill');
            bars.forEach(bar => {{
                const width = bar.style.width;
                bar.style.width = '0%';
                setTimeout(() => {{ bar.style.width = width; }}, 100);
            }});
        }});
    </script>
</body>
</html>"""
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        print(f"\n HTML raporu oluşturuldu: {output_path}")
        return output_path


def interactive_menu():
    """İnteraktif menü göster"""
    print("\n" + "="*70)
    print("GELİŞMİŞ STEGANOGRAFİ ANALİZ ARACI")
    print("="*70 + "\n")
    
    print("Lütfen analiz yöntemini seçin:\n")
    print("  1  LSB (Least Significant Bit) Analizi")
    print("  2  DCT (Discrete Cosine Transform) Analizi - JPEG için")
    print("  3  Histogram Analizi")
    print("  4  RS (Regular-Singular) Analizi")
    print("  5  Sample Pair Analizi")
    print("  6  Entropi Analizi")
    print("  7  Machine Learning Tabanlı Tespit")
    print("  8  TÜM ANALİZLER (Önerilen)")
    print("  9  Batch Analiz (Çoklu dosya)")
    print("  0  Çıkış")
    
    choice = input("\n Seçiminiz (1-9): ").strip()
    return choice


def batch_analysis(pattern, methods):
    """Batch analiz - çoklu dosya"""
    print(f"\n{'='*70}")
    print(f"BATCH ANALİZ - Pattern: {pattern}")
    print(f"{'='*70}\n")
    
    files = glob.glob(pattern)
    
    if not files:
        print(f" Pattern ile eşleşen dosya bulunamadı: {pattern}")
        return
    
    print(f" {len(files)} dosya bulundu\n")
    
    results_summary = []
    
    for idx, filepath in enumerate(files, 1):
        print(f"\n[{idx}/{len(files)}] Analiz ediliyor: {os.path.basename(filepath)}")
        print("-" * 70)
        
        try:
            tool = AdvancedSteganalysisTool(filepath, methods)
            result = tool.analyze()
            
            # Özet bilgi
            ml_score = result['ml_prediction'].get('score', 0) if result['ml_prediction'] else 0
            findings_count = len(result['suspicious_findings'])
            
            results_summary.append({
                'file': os.path.basename(filepath),
                'ml_score': ml_score,
                'findings': findings_count,
                'status': ' ŞÜPHELİ' if ml_score > 50 or findings_count > 3 else ' TEMİZ'
            })
            
            # HTML raporu
            report_name = f"batch_report_{Path(filepath).stem}.html"
            tool.generate_html_report(report_name)
            
        except Exception as e:
            print(f" Hata: {e}")
            results_summary.append({
                'file': os.path.basename(filepath),
                'ml_score': 0,
                'findings': 0,
                'status': ' HATA'
            })
    
    # Özet rapor
    print(f"\n\n{'='*70}")
    print(" BATCH ANALİZ ÖZETİ")
    print(f"{'='*70}\n")
    
    print(f"{'Dosya':<30} {'ML Skor':<10} {'Bulgular':<10} {'Durum'}")
    print("-" * 70)
    
    for result in results_summary:
        print(f"{result['file']:<30} {result['ml_score']:>7.1f}   {result['findings']:>7}   {result['status']}")
    
    # İstatistikler
    suspicious_count = sum(1 for r in results_summary if '🚨' in r['status'])
    print(f"\n İstatistikler:")
    print(f"   Toplam dosya: {len(files)}")
    print(f"   Şüpheli: {suspicious_count}")
    print(f"   Temiz: {len(files) - suspicious_count}")


def main():
    parser = argparse.ArgumentParser(
        description='Gelişmiş Steganografi Analiz Aracı',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Örnekler:
  # İnteraktif mod
  %(prog)s
  
  # Tek dosya - tüm analizler
  %(prog)s image.png
  
  # Belirli analizler
  %(prog)s image.png --methods lsb histogram entropy
  
  # Batch analiz
  %(prog)s --batch "*.png" --methods all
  
  # HTML raporu
  %(prog)s image.png --html report.html
        """
    )
    
    parser.add_argument('filepath', nargs='?', help='Analiz edilecek görüntü dosyası')
    parser.add_argument('--methods', '-m', nargs='+', 
                       choices=['lsb', 'dct', 'histogram', 'rs', 'sample_pair', 'entropy', 'ml', 'all'],
                       default=['all'],
                       help='Analiz yöntemleri')
    parser.add_argument('--batch', '-b', metavar='PATTERN',
                       help='Batch analiz için dosya pattern (örn: "*.png")')
    parser.add_argument('--html', '--report', metavar='OUTPUT',
                       help='HTML raporu oluştur')
    parser.add_argument('--interactive', '-i', action='store_true',
                       help='İnteraktif menü modu')
    
    args = parser.parse_args()
    
    # İnteraktif mod veya argüman yoksa
    if args.interactive or (not args.filepath and not args.batch):
        while True:
            choice = interactive_menu()
            
            if choice == '0':
                print("\n Çıkılıyor...\n")
                sys.exit(0)
            
            if choice == '9':
                pattern = input("\n Dosya pattern (örn: *.png): ").strip()
                method_choice = input("Analiz yöntemi (1-8, 8=tümü): ").strip()
                
                method_map = {
                    '1': ['lsb'], '2': ['dct'], '3': ['histogram'],
                    '4': ['rs'], '5': ['sample_pair'], '6': ['entropy'],
                    '7': ['ml'], '8': ['all']
                }
                
                methods = method_map.get(method_choice, ['all'])
                batch_analysis(pattern, methods)
                
                input("\n Devam etmek için Enter'a basın...")
                continue
            
            filepath = input("\n Dosya yolu: ").strip()
            
            if not os.path.exists(filepath):
                print(f"\n Dosya bulunamadı: {filepath}")
                input("Devam etmek için Enter'a basın...")
                continue
            
            method_map = {
                '1': ['lsb'], '2': ['dct'], '3': ['histogram'],
                '4': ['rs'], '5': ['sample_pair'], '6': ['entropy'],
                '7': ['ml'], '8': ['all']
            }
            
            methods = method_map.get(choice, ['all'])
            
            try:
                tool = AdvancedSteganalysisTool(filepath, methods)
                tool.analyze()
                
                html_choice = input("\n HTML raporu oluşturulsun mu? (e/h): ").strip().lower()
                if html_choice == 'e':
                    html_name = f"report_{Path(filepath).stem}.html"
                    tool.generate_html_report(html_name)
                    print(f"\n HTML raporu: {html_name}")
                
            except Exception as e:
                print(f"\n Hata: {e}")
            
            input("\n Devam etmek için Enter'a basın...")
    
    # Batch mod
    elif args.batch:
        batch_analysis(args.batch, args.methods)
    
    # Normal mod
    elif args.filepath:
        if not os.path.exists(args.filepath):
            print(f" Hata: '{args.filepath}' dosyası bulunamadı!")
            sys.exit(1)
        
        try:
            tool = AdvancedSteganalysisTool(args.filepath, args.methods)
            tool.analyze()
            
            if args.html:
                tool.generate_html_report(args.html)
                print(f"\n HTML raporu: {args.html}")
                
        except Exception as e:
            print(f" Hata: {e}")
            sys.exit(1)


if __name__ == '__main__':
    main()
